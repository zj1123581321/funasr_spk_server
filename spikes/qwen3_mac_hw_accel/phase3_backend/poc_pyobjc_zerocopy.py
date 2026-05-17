"""
PoC: PyObjC zero-copy 绕过 SIGSEGV (Gemini deep research C.6 推荐).

假设 (Gemini): 用 MLMultiArray.initWithDataPointer 创建 zero-copy multi-array,
MLMultiArray 只持 numpy 裸 C 指针, 不 retain Python 对象。
后台 lingering reset 触发 dealloc 时, ObjC 释放它自己的 wrapper,
不调 _PyObject_Free → 不踩 GIL race。

反驳 (Kimi): 即使用 PyObjC, MLFeatureValue 内部 NSDictionary 仍可能持 Python ref。

实测决定。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


# CoreML constants (MLMultiArray.h)
MLMultiArrayDataTypeFloat32 = 0x10020
MLMultiArrayDataTypeInt32 = 0x20020
# MLComputeUnits enum
MLComputeUnitsCPUAndNeuralEngine = 2
MLComputeUnitsCPUOnly = 0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlpackage", required=True)
    ap.add_argument("--time", type=int, default=520)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=20, help="先跑 N 次 predict 注册 lingering buffer")
    ap.add_argument("--wait-lingering", type=int, default=120,
                    help="跑完 predict 后等多少秒 (之前观察 ~95s lingerTime)")
    ap.add_argument("--with-sherpa", action="store_true",
                    help="加载 sherpa-onnx 加速 race 触发")
    ap.add_argument("--embedding", default="models/qwen3_diarize/sherpa/nemo-titanet-small/embedding.onnx")
    return ap.parse_args()


def main():
    args = parse_args()

    import objc
    print(f"pyobjc {objc.__version__}", flush=True)
    objc.loadBundle("CoreML", bundle_path="/System/Library/Frameworks/CoreML.framework", module_globals=globals())
    from Foundation import NSURL, NSNumber

    MLModel = objc.lookUpClass("MLModel")
    MLModelConfiguration = objc.lookUpClass("MLModelConfiguration")
    MLMultiArray = objc.lookUpClass("MLMultiArray")
    MLDictionaryFeatureProvider = objc.lookUpClass("MLDictionaryFeatureProvider")
    MLFeatureValue = objc.lookUpClass("MLFeatureValue")

    # ----- 1. compile mlpackage → mlmodelc, 加载 with CPU_AND_NE -----
    p = os.path.abspath(args.mlpackage)
    url = NSURL.fileURLWithPath_(p)
    print(f"[1.1] compileModelAtURL ...", flush=True)
    t0 = time.time()
    compiled_url = MLModel.compileModelAtURL_error_(url, None)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    config = MLModelConfiguration.alloc().init()
    config.setComputeUnits_(MLComputeUnitsCPUAndNeuralEngine)
    print(f"[1.2] load model with CPU_AND_NE ...", flush=True)
    t0 = time.time()
    mlmodel = MLModel.modelWithContentsOfURL_configuration_error_(compiled_url, config, None)
    print(f"  done in {time.time()-t0:.1f}s, model={mlmodel is not None}", flush=True)
    assert mlmodel is not None

    # ----- 2. 准备 numpy buffers (zero-copy) -----
    h = np.ascontiguousarray(np.random.randn(1, args.time, args.dim).astype(np.float32))
    m = np.ascontiguousarray(np.ones((1, args.time), dtype=np.int32))

    def np_to_mlarray(arr, mltype):
        """Zero-copy numpy → MLMultiArray via initWithDataPointer.

        重点: deallocator=None → MLMultiArray 不释放 buffer, numpy 自己管。
        必须保 numpy ref 在 predict 期间存活 (上层 caller 责任)。
        """
        shape = [NSNumber.numberWithLong_(s) for s in arr.shape]
        strides = [NSNumber.numberWithLong_(s // arr.itemsize) for s in arr.strides]
        ml = MLMultiArray.alloc().initWithDataPointer_shape_dataType_strides_deallocator_error_(
            arr.ctypes.data, shape, mltype, strides, None, None,
        )
        if ml is None:
            raise RuntimeError("MLMultiArray init failed")
        return ml

    # ----- (Optional) 加载 sherpa-onnx 加速 race 触发 -----
    if args.with_sherpa:
        print(f"[1.3] load sherpa-onnx (加速 race)", flush=True)
        import sherpa_onnx
        sherpa_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=args.embedding, num_threads=4, provider="cpu", debug=False,
        )
        assert sherpa_cfg.validate()
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_cfg)
        print(f"  sherpa loaded", flush=True)

    # ----- 3. 跑 N 次 predict (注册 lingering buffer) -----
    print(f"[3] {args.runs} predict 注册 lingering buffer", flush=True)
    t_total = time.time()
    for i in range(args.runs):
        ml_h = np_to_mlarray(h, MLMultiArrayDataTypeFloat32)
        ml_m = np_to_mlarray(m, MLMultiArrayDataTypeInt32)
        feat_dict = {"hidden_states": ml_h, "key_padding_mask": ml_m}
        provider = MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(feat_dict, None)
        assert provider is not None

        t0 = time.time()
        pred = mlmodel.predictionFromFeatures_error_(provider, None)
        assert pred is not None
        out_mla = pred.featureValueForName_("last_hidden_state").multiArrayValue()
        if i == 0:
            shape_list = [int(out_mla.shape()[k]) for k in range(out_mla.shape().count())]
            print(f"  predict[0]: {time.time()-t0:.2f}s, out shape={shape_list}", flush=True)

        if args.with_sherpa and i % 2 == 0:
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=16000, waveform=np.zeros(48000, np.float32))
            stream.input_finished()
            _ = extractor.compute(stream)

    print(f"[3] {args.runs} predicts done in {time.time()-t_total:.1f}s", flush=True)

    # ----- 4. 等 lingering 超时 -----
    print(f"[4] 等 {args.wait_lingering}s lingering reset (Apple lingerTime 实测 ~40-95s)", flush=True)
    for s in range(args.wait_lingering):
        time.sleep(1)
        if (s + 1) % 10 == 0:
            print(f"  +{s+1}s ... 仍活着", flush=True)

    print(f"\n✅ PoC PASS: PyObjC zero-copy + {args.runs} predict + 等 {args.wait_lingering}s 无 SIGSEGV")


if __name__ == "__main__":
    main()
