# Deep research prompt — CoreML MLE5ExecutionStream lingering reset + Python framework SIGSEGV

> 使用方式: 把整份内容粘到 Gemini Deep Research / Kimi Deep Search,要求"对每一个 §A-D 问题分别给答案 + 引用来源链接 + 给可立即 reproducible 的最小代码示例"。

## 0. 我们的问题(背景)

- 平台:macOS 26.5 (build 25F71, 2026-05),Apple M1 Max,Python 3.12.13
- 软件:`coremltools 9.0`,`torch 2.8.0`(用于 export),`onnxruntime 1.26`,`sherpa-onnx`(自带 libonnxruntime 1.24.4)
- 模型:Qwen3-ASR encoder backend(.mlpackage,FP16,static shape `(1, 520, 1024)`,1.7B 模型的 audio_tower 部分,317.5M params,583 MB)
- 调用方式:Python 主线程 `mlmodel.predict({hidden_states: np.float32, key_padding_mask: np.int32})`
- 加载:`ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)`,cold start ANE plan compile ~24s,warm 69 ms/run

### 症状(完全 reproducible,5 份独立 .ips 同 stack)

```
EXC_BAD_ACCESS (SIGSEGV) at 0x0000000000000010, KERN_INVALID_ADDRESS
Faulting thread (libdispatch worker, NOT main):

  [Python]                 _PyObject_Free
  [libcoremlpython.so]     ?
  [libobjc.A.dylib]        object_cxxDestructFromClass
  [libobjc.A.dylib]        objc_destructInstance_nonnull_realized
  [libobjc.A.dylib]        _objc_rootDealloc
  [CoreML]                 -[MLFeatureValue dealloc]
  [CoreML]                 -[MLE5InputPortBinder reset]
  [CoreML]                 -[MLE5InputPort reset]
  [CoreML]                 -[MLE5ExecutionStreamOperation reset]
  [CoreML]                 -[MLE5ExecutionStream _reset]
  [CoreML]                 __43-[MLE5ExecutionStream resetAfterLingering:]_block_invoke
  [libdispatch.dylib]      _dispatch_client_callout
  [libdispatch.dylib]      _dispatch_continuation_pop
  [libdispatch.dylib]      _dispatch_source_latch_and_call
  [libdispatch.dylib]      _dispatch_source_invoke
  [libdispatch.dylib]      _dispatch_lane_serial_drain
  [libdispatch.dylib]      _dispatch_lane_invoke
  [libdispatch.dylib]      _dispatch_root_queue_drain_deferred_wlh
  [libdispatch.dylib]      _dispatch_workloop_worker_thread
  [libsystem_pthread.dylib] _pthread_wqthread
```

### 触发链(已实测)

1. 加载 mlpackage (`MLModel(path, compute_units=CPU_AND_NE)`),ANE plan compile 24s
2. 预热推理 1 次(`predict(dummy_input)`),input MLFeatureValue 被 CoreML 内部加入 lingering 池
3. **几秒后**(`lingerTime`,不可控,从 ~40s 到 ~95s 不等,跨 macOS 26.0/26.5 行为差异)
4. `MLE5ExecutionStream.resetAfterLingering:` 触发 dispatch_async block
5. **libdispatch worker thread**(非 main,不持 GIL)进入 block
6. `-[MLFeatureValue dealloc]` 内部 `objc release` 链 → `libcoremlpython.so` 的 C++ destructor
7. C++ destructor 调 `_PyObject_Free` 释放 numpy array 引用
8. `_PyObject_Free` 假设 GIL 持有,访问 `0x10` (Python tstate 偏移) → SIGSEGV

### 关键观察

- **单 audio 短跑通**(parity test cos=0.999069),因为 process 在 lingerTime 内结束,reset 没触发
- **完整 pipeline 100% 崩**,因为 mlpackage 加载后 → 同 process 加载 sherpa-onnx → sherpa 启动它自己的 libdispatch workers,这些 workers 跟 MLE5 lingering worker 抢资源,**或者只是因为时间够长 lingerTime 到了**(单纯 mlpackage 加载后 sleep 5 min 也会崩?未测,值得求证)
- 试过 `gc.collect() + time.sleep(3)`:无效
- 试过 `inputs = {k: v.copy()}` 让 MLFeatureValue 持有独立 buffer:无效(MLFeatureValue 内部仍 retain Python ref)
- 试过 N=1 单 worker 排除并发:仍崩(是单进程内 race,跟并发无关)

### Apple 升级测试结果

| macOS version | crash time | stack |
|---|---|---|
| 26.0 Beta (25C56) | ~40s | 同 |
| 26.5 正式 (25F71) | ~95s | 完全同 |

26.0 → 26.5 升级**没修**,只是 `lingerTime` 调大了(推测)。

---

## §A — coremltools / CoreML 是否有官方 sync / disable lingering API?

**核心问题**:`coremltools.models.MLModel.predict()` 或底层 `MLModel.predictionFromFeatures:options:error:` 是否有任何方式禁用 `MLE5ExecutionStream` 的 `lingerTime` / `resetAfterLingering:` 异步行为?

请逐条调研:

1. **coremltools >= 9.0 文档 / changelog**:`MLModel.predict()` 有没有 `synchronous=`、`lingerTime=`、`releaseInputs=` 之类参数?
2. **coremltools.models.model.MLModel.__init__** 接受的所有 kwargs(尤其是 `compute_units` 之外的隐藏参数)
3. **`MLModelConfiguration`**(ObjC API)/ `MLPredictionOptions`:
   - `MLPredictionOptions.usesCPUOnly` (deprecated)
   - `MLPredictionOptions.outputBackings` (iOS 16+) — 是否影响 lingering
   - 是否有 `releaseOutputsOnReturn` / `synchronousExecution` flag
4. **`MLModelAsset`**(iOS 18 新 API)是否绕开 MLE5 stream
5. 是否能通过 ObjC runtime swizzle `-[MLE5ExecutionStream resetAfterLingering:]` 改成同步执行
6. coremltools GitHub issues + apple/coremltools discussions 搜索:`lingering`、`resetAfter`、`MLE5`、`SIGSEGV`、`PyObject_Free`、`libdispatch dealloc race`
7. **环境变量 / plist key** 控制 lingering 行为?
   - `MLE5_DISABLE_LINGERING_RESET`?(瞎猜)
   - `CoreMLConfigStaticAnalysis` / `MLComputeDevice` 系列私有 env
   - `defaults read com.apple.CoreML` 可见的 key
   - 通过 `dyld -e ... predict.py` 看 framework 引用哪些 env

请给可立即 reproducible 的 **API 名字 + Python 调用示例** 或者**确认 "目前没有官方 API"** 的证据。

---

## §B — 业界其他 CoreML + 多 framework 大模型 production 案例怎么解的?

**核心问题**:WhisperKit / ParakeetASR / Ivan Mosin 的 qwen3-asr-swift / argmaxinc speech-swift / 任何 production-grade CoreML + sherpa-onnx 或 ONNX Runtime 共存的项目,有没有踩过同样 `MLE5ExecutionStream resetAfterLingering` + libdispatch worker 无 GIL 的 SIGSEGV?

逐项:

1. **WhisperKit (argmaxinc/WhisperKit)** 是 Swift 项目,不踩 Python GIL。但他们 README / blog 有没有提到 lingerTime / stream reset 类问题?
2. **ParakeetASR (argmaxinc)** — INT4 ANE,Swift,但他们的 `predict` wrapper 怎么管理 input 生命周期?
3. **Ivan Mosin "Apple Neural Engine reverse engineering" 系列博文** — 他踩过哪些坑?具体到 `MLFeatureValue` 生命周期管理
4. **coremltools issues**:
   - 搜 "SIGSEGV"、"segfault"、"_PyObject_Free"、"MLE5"、"dispatch worker"
   - 关键 issue 编号 + 状态(open / closed / wontfix)
5. **Python framework 共存案例**:有没有 production 项目同时用 `coremltools.MLModel` + `sherpa-onnx`(或任何 libdispatch-using lib)?他们怎么解决?
6. **Apple Developer Forums**:搜 "MLFeatureValue dealloc dispatch" / "MLModel SIGSEGV Python"
7. **Reddit r/MLQuestions、r/swift、r/CoreML** 类似 case
8. coremltools 上是否有 "MLModelHotPath" / "RunInBackground" 类 sample code 处理 dealloc 同步?

请给:**至少 3 个独立来源的 evidence**(GitHub issue / blog / Apple forum / Stack Overflow),最好附**有人实际解决过的代码示例**。

---

## §C — 不改 coremltools / 不换 CoreML 的低成本绕过方案

**核心问题**:如果 coremltools 和 macOS 都没法改,应用层有什么干净的隔离方式?

请评估每一个方案的:**可行性 / 工程量 / 性能开销 / 副作用**:

### C.1 — 子进程隔离 mlpackage backend

把 backend `MLModel` 加载到独立 Python 子进程,主进程用 `multiprocessing.Pipe` 或 `unix domain socket` 传 numpy。

- 每次 backend predict 多 1-3 ms IPC(estimate)
- 子进程 mlpackage 单独驻留 1GB RSS
- N=2 worker:2 主 + 2 子 = 4 进程
- **问题**:子进程内部 sherpa-onnx 不加载,libdispatch worker 不抢占 — 真的能避开吗?还是 mlpackage **自己内部**的 libdispatch 就足以触发,跟 sherpa 无关?
- coremltools shared memory transport(`ct.utils.shared_memory_transport`?)是否存在?

### C.2 — PyObjC `autorelease_pool()` + 同步主动 reset

```python
import objc
with objc.autorelease_pool():
    pred = mlmodel.predict(inputs)
    out = pred["last_hidden_state"].copy()
# autorelease_pool 退出时强制释放?
```

- 但 `resetAfterLingering` 是 `dispatch_async` 调度,跟 autorelease 无关
- 能否用 PyObjC 直接访问 `mlmodel._proxy._model._stream` 调 `_reset`?ObjC private API

### C.3 — 主动触发 lingering reset 在主线程下

每次 predict 后,**主线程**调一次特殊 op(零 input)强迫 stream 复用,绕开异步 lingering。

- 是否可行?coremltools 内部 stream 是否每次 predict 复用 vs 新建?

### C.4 — Free-threaded Python 3.13+ (no GIL)

`python3.13t` GIL-less 版本下,后台 thread 调 `_PyObject_Free` 不会 segfault。

- coremltools 9.0 兼容 Python 3.13+?
- numpy 2.x 兼容?(我们 stack 用 torch 2.8 + onnxruntime 1.26)
- 性能 regression?

### C.5 — 单纯 import 顺序 / 加载顺序调整

- 先加载 sherpa-onnx → 再加载 mlpackage → 跑 predict
- 反过来:先跑完所有 ANE predict → del mlmodel → 再加载 sherpa
- 实测看 lifecycle 谁先谁后

### C.6 — 用 PyObjC 直接调 MLModel,不走 libcoremlpython.so

```python
from CoreML import MLModel as MLModelObjC, MLDictionaryFeatureProvider, MLMultiArray
model = MLModelObjC.modelWithContentsOfURL_error_(url, None)
features = MLDictionaryFeatureProvider.dictionaryFeatureProviderWithDictionary_error_(...)
pred = model.predictionFromFeatures_error_(features, None)
```

- 绕开 coremltools Python binding 这一层
- 是否仍踩同 race?(根因在 CoreML.framework,不在 coremltools)
- 工程量 / 性能差异

请对每个 C.x **明确给出"会 work / 不会 work / 不确定" + 理由**。如果 unknown,给一个 30-min reproducer 让我自己试。

---

## §D — 编译时(转换时) workaround

**核心问题**:能不能让 `ct.convert(...)` 产出的 mlpackage **不走 MLE5 后端**?

1. **`convert_to="neuralnetwork"`** 而不是 `mlprogram`
   - 我们 PoC 已证 NeuralNetwork format silent fallback CPU 无收益
   - 但若 fallback 是 GPU 不是 CPU,值不值得二次验证?
2. **`compute_precision=FLOAT32`** 是否影响 stream 选择?
3. **`minimum_deployment_target=macOS14`** (降级到 ML Program v1) 是否走老 stream 实现?
4. **降级 `coremltools 8.x`**(8.1 / 8.2)export 出来的 mlpackage 在 macOS 26.5 上跑,行为是否不同?
   - coremltools 8.x release notes 关于 MLE5
   - 是否 v8 还没引入 lingering 优化

请给:**每个 option 的预期模型大小变化 / 推理速度变化 / 是否能避开 MLE5**。

---

## 输出格式

请在最终报告里包含:

```
# §A 报告
- A.1 ... 答案/证据/示例
- A.2 ...
...

# §B 报告
- B.1 ...
...

# §C 报告 (对每个 C.x 评估)
| 方案 | 可行性 | 工程量 | 性能 | 副作用 | 我的推荐 |
|------|--------|--------|------|--------|----------|
| C.1  | ...    | ...    | ...  | ...    | ...      |
...

# §D 报告
...

# 最终推荐路径
按优先级给 1-3 个最值得我先试的方向,每个给:
- 预期工程量 (小时)
- 成功概率 (主观 %)
- 失败 fallback
```

如果某个 § 的某个子问题没找到答案,**明确说 "未找到"**,不要瞎编。

---

## 不需要研究的(已确认负面)

- ❌ `MLComputeUnits=ALL`(抢 llama.cpp Metal)
- ❌ `ModelFormat=NeuralNetwork` + units `CPUAndNeuralEngine`(Phase 2 已证 silent fallback CPU)
- ❌ 反向 un-fuse 现有 583MB ONNX(Gemini + Kimi 上次 deep research 已否)
- ❌ MLX 重写(跟 llama.cpp 抢 Metal dispatch queue)
- ❌ `gc.collect() + time.sleep(3)` 等 lingering 自然完成(实测无效)
- ❌ `inputs = {k: v.copy()}` 防御性拷贝(实测无效)
- ❌ 等 macOS 26.5 正式版修(实测仍崩)
- ❌ 减少 N(N=1 也崩,跟并发无关)

## 参考资料

- 完整 spike 报告:`spikes/qwen3_mac_hw_accel/phase3_backend_mlpackage.md`
- 5 份 SIGSEGV crash report(.ips JSON 格式):`~/Library/Logs/DiagnosticReports/Python-2026-05-17-{012628,012701,012908,013428,013705,101250}.ips`
- 已落地代码:`src/core/vendor/qwen_asr_gguf/inference/encoder.py` 的 `COREML_ANE_FULL` 分支
- Ivan Mosin 文章:`docs/ref/onnx_ane/gemini.md` + `docs/ref/onnx_ane/kimi.md`(上次 Phase 3 deep research 输出)
