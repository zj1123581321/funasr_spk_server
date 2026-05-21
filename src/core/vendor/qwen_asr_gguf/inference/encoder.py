# coding=utf-8
import os
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort


class FastWhisperMel:
    """基于 NumPy 的纯净版 Mel 提取器 (彻底干掉 librosa 的 numba JIT 启动延时)"""
    def __init__(self, filter_path: str = None, n_mels=128, sr=16000, n_fft=400, f_min=0, f_max=8000, norm="slaney", mel_scale="slaney"):
        self.n_fft = n_fft
        self.hop_length = 160
        self.n_mels = n_mels
        
        if filter_path and os.path.exists(filter_path):
            self.filters = np.load(filter_path)
        else:
            self.filters = self._generate_filters(sr, n_fft, n_mels, f_min, f_max, norm, mel_scale)
            
        # 提前计算并缓存好汉明窗 (Qwen3/Whisper/Librosa 使用 Hann 窗)
        self.window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n_fft) / self.n_fft)
        
    def _generate_filters(self, sr, n_fft, n_mels, f_min, f_max, norm, mel_scale):
        """
        生成组件化的梅尔滤波器组 (兼容 torchaudio 行为)
        norm: "slaney" (面积归一化) 或 None
        mel_scale: "slaney" (分段线性+对数) 或 "htk" (纯对数)
        """
        def hz_to_mel(freq, scale):
            if scale == "htk":
                return 2595.0 * np.log10(1.0 + (freq / 700.0))
            # Slaney Scale (Linear + Log)
            f_min_sl, f_sp_sl = 0.0, 200.0 / 3
            mels = (freq - f_min_sl) / f_sp_sl
            min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
            min_log_mel = (min_log_hz - f_min_sl) / f_sp_sl
            if isinstance(freq, np.ndarray):
                mask = freq >= min_log_hz
                mels[mask] = min_log_mel + np.log(freq[mask] / min_log_hz) / logstep
            elif freq >= min_log_hz:
                mels = min_log_mel + np.log(freq / min_log_hz) / logstep
            return mels

        def mel_to_hz(mels, scale):
            if scale == "htk":
                return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
            # Slaney Scale (Linear + Log)
            f_min_sl, f_sp_sl = 0.0, 200.0 / 3
            freqs = f_min_sl + f_sp_sl * mels
            min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
            min_log_mel = (min_log_hz - f_min_sl) / f_sp_sl
            if isinstance(mels, np.ndarray):
                mask = mels >= min_log_mel
                freqs[mask] = min_log_hz * np.exp(logstep * (mels[mask] - min_log_mel))
            elif mels >= min_log_mel:
                freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
            return freqs

        n_freqs = n_fft // 2 + 1
        all_freqs = np.linspace(0, sr // 2, n_freqs)
        m_pts = np.linspace(hz_to_mel(f_min, mel_scale), hz_to_mel(f_max, mel_scale), n_mels + 2)
        f_pts = mel_to_hz(m_pts, mel_scale)
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts[np.newaxis, :] - all_freqs[:, np.newaxis]
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
        up_slopes = slopes[:, 2:] / f_diff[1:]
        fb = np.maximum(0, np.minimum(down_slopes, up_slopes))
        
        # Area Normalization
        if norm == "slaney":
            enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
            fb *= enorm[np.newaxis, :]
            
        return fb.astype(np.float32)
        
    def __call__(self, audio: np.ndarray, dtype=np.float32) -> np.ndarray:
        # 1. Padding (与 librosa 的 center=True 行为保持一致)
        pad_len = int(self.n_fft // 2)
        y = np.pad(audio, pad_len, mode='reflect')
        
        # 2. 高效分帧 (利用 numpy 内存视图，耗时几乎为0)
        num_frames = 1 + (len(y) - self.n_fft) // self.hop_length
        shape = (self.n_fft, num_frames)
        strides = (y.itemsize, self.hop_length * y.itemsize)
        frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
        
        # 3. 加窗并执行实数 FFT
        stft_res = np.fft.rfft(frames * self.window[:, np.newaxis], axis=0)
        
        # 4. 能量谱
        magnitudes = np.abs(stft_res) ** 2
        
        # 5. Mel 映射
        mel_spec = np.dot(self.filters.T, magnitudes)
        
        # 6. 取对数
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        
        # 7. 归一化
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        
        # 8. 帧对齐：丢弃多余帧
        n_frames_out = audio.shape[-1] // self.hop_length
        log_spec = log_spec[:, :n_frames_out]
        
        return log_spec.astype(dtype)

def get_feat_extract_output_lengths(input_lengths):
    """
    完全复刻官方 Qwen3 前端逻辑，计算最终有效的输出帧数。
    用于从拼接好的 (N*13) 结果中切出有效部分。
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return int(output_lengths)

class QwenAudioEncoder:
    """Qwen3 音频编码器 (Split Frontend + Backend)"""
    def __init__(self, frontend_path: str, backend_path: str, onnx_provider: str = 'CPU', dml_pad_to: int = 30, verbose: bool = True):
        self.verbose = verbose
        self.onnx_provider = onnx_provider.upper()
        self.active_dml = False
        self.dml_pad_to = dml_pad_to
        # 预计算目标长度：每 1 秒对应 13 帧 hidden_states
        self.h_target_len = self.dml_pad_to * 13
        # backend mlpackage MLModel 句柄 (仅 COREML_ANE_FULL 启用), None 表示走 ONNX backend
        self.sess_be_mlmodel = None
        # mlpackage 走静态形状 T=h_target_len, _run_backend 走 padding 路径
        self.active_static_be = False

        # 初始化 ONNX Session Options
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        providers = ['CPUExecutionProvider']
        # frontend / backend 默认共用 providers; COREML_ANE_FE 会分开覆盖
        providers_fe = providers
        providers_be = providers
        load_backend_as_mlpackage = False  # COREML_ANE_FULL 触发

        if self.onnx_provider in ('TRT', 'TENSORRT') and 'TensorrtExecutionProvider' in available_providers:
            providers.insert(0, ('TensorrtExecutionProvider', {
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': Path(backend_path).parent / 'trt_cache',
            }))
        elif self.onnx_provider == 'DML' and 'DmlExecutionProvider' in available_providers:
            providers.insert(0, 'DmlExecutionProvider')
            self.active_dml = True
        elif self.onnx_provider == 'CUDA' and 'CUDAExecutionProvider' in available_providers:
            providers.insert(0, 'CUDAExecutionProvider')
        elif self.onnx_provider in ('COREML_ANE_FE', 'COREML_ANE_FULL'):
            # macOS + Apple Silicon CoreML 路径
            # - COREML_ANE_FE   : frontend 走 CoreML ANE, backend 仍 ONNX CPU (Phase 2)
            # - COREML_ANE_FULL : frontend ANE + backend mlpackage ANE        (Phase 3)
            # PoC 验证 (M1 Max):
            #   - Phase 2 FE 单变量 -7.5%, 跟 num_threads=4 组合 -16.1%
            #   - Phase 3 backend mlpackage cos 0.999069 max_abs 4.58e-3, warm 69ms/run
            # 反例排查:
            #   - units='ALL' 抢 llama.cpp Metal -> llm_decode +73s
            #   - fmt='NeuralNetwork' silent fallback CPU 无收益
            #   - backend ONNX 走 CoreML EP 卡 axis 4 op 兼容报错 (改用 mlpackage 绕过)
            # 详见 spikes/qwen3_mac_hw_accel/{coreml_asr_encoder.md,phase3_backend/}
            import sys as _sys
            if _sys.platform != 'darwin':
                if self.verbose:
                    print(f"--- [Encoder] {self.onnx_provider} 仅 macOS 支持, fallback CPU ---")
            elif 'CoreMLExecutionProvider' not in available_providers:
                if self.verbose:
                    print(f"--- [Encoder] onnxruntime 未编 CoreMLExecutionProvider, {self.onnx_provider} fallback CPU ---")
            else:
                providers_fe = [
                    ('CoreMLExecutionProvider', {
                        'ModelFormat': 'MLProgram',
                        'MLComputeUnits': 'CPUAndNeuralEngine',
                        'RequireStaticInputShapes': '0',
                        'EnableOnSubgraphs': '0',
                    }),
                    'CPUExecutionProvider',
                ]
                providers_be = ['CPUExecutionProvider']
                if self.onnx_provider == 'COREML_ANE_FULL':
                    # backend 路径必须是 .mlpackage 目录; 否则降级 COREML_ANE_FE 行为
                    be_p = Path(backend_path)
                    if be_p.suffix == '.mlpackage' and be_p.is_dir():
                        load_backend_as_mlpackage = True
                    else:
                        if self.verbose:
                            print(
                                f"--- [Encoder] COREML_ANE_FULL: backend_path 不是 .mlpackage 目录, "
                                f"降级 COREML_ANE_FE (frontend ANE + backend ONNX CPU) ---"
                            )

        if self.verbose:
            be_label = '.mlpackage (CoreML ANE)' if load_backend_as_mlpackage else providers_be[0]
            print(f"--- [Encoder] 加载 Split 模型 (FE: {providers_fe[0]}, BE: {be_label}, Pad: {dml_pad_to}s) ---")
            print(f"    Frontend: {os.path.basename(frontend_path)}")
            print(f"    Backend:  {os.path.basename(backend_path)}")

        # 加载 frontend (ONNX, 必有)
        self.sess_fe = ort.InferenceSession(frontend_path, sess_options=sess_opts, providers=providers_fe)
        if load_backend_as_mlpackage:
            # COREML_ANE_FULL: backend 走 PyObjC zero-copy CoreML runner + CPU_AND_NE
            # 加载 ~24-40s (cold ANE plan compile); _run_backend 走 static pad (h_target_len)
            #
            # 为什么用 PyObjC 而不用 coremltools.models.MLModel?
            # coremltools 的 pybind11 binding 在 MLE5ExecutionStream 后台 lingering reset
            # 时调 _PyObject_Free 但不持 GIL, 跟 sherpa-onnx 共存会 SIGSEGV (macOS 26+).
            # PyObjC zero-copy (initWithDataPointer + deallocator=None) 让 MLMultiArray
            # 只持 numpy 裸 C 指针, dealloc 不调 Python C API, 绕开 race.
            # 详见 spikes/qwen3_mac_hw_accel/phase3_backend/poc_pyobjc_zerocopy.py
            from .coreml_runner import CoreMLZeroCopyRunner
            # env override: 默认 CPU_AND_NE (ANE), 但 N=2 时 ANE 跟 frontend 4 路冲突可能触发
            # llama.cpp ggml_abort (N=1 work, N=2 fail). 设 FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS=CPU_AND_GPU
            # 让 backend 走 Metal GPU, frontend 独占 ANE.
            # 注意: Phase 2 警告 units=ALL 抢 llama.cpp Metal, 这里 backend mlpackage CPU_AND_GPU
            # 跟 llama.cpp 同 Metal dispatch queue 可能 llm_decode 退化, 实测.
            import os as _os
            be_units = _os.environ.get("FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS", "CPU_AND_NE")
            self.sess_be_mlmodel = CoreMLZeroCopyRunner(
                backend_path,
                compute_units=be_units,
                verbose=self.verbose,
            )
            self.active_static_be = True
            self.sess_be = None  # 跟 ONNX session 区分开
        else:
            self.sess_be = ort.InferenceSession(backend_path, sess_options=sess_opts, providers=providers_be)
        
        self.mel_extractor = FastWhisperMel()
        
        # 检测精度 (以前端为准)
        try:
            fe_input_type = self.sess_fe.get_inputs()[0].type
            self.input_dtype = np.float16 if 'float16' in fe_input_type else np.float32
        except:
            self.input_dtype = np.float32

        # 预热处理
        if self.dml_pad_to > 0 and self.active_dml:
            if self.verbose: print(f"--- [Encoder] 正在预热 (固定形状: {self.dml_pad_to}s)... ---")
            dummy_wav = np.zeros(int(16000 * self.dml_pad_to)).astype(np.float32)
            _ = self.encode(dummy_wav)
        elif self.sess_be_mlmodel is not None:
            # COREML_ANE_FULL: 预热必须 padding 到 h_target_len (mlpackage static shape)
            if self.verbose: print(f"--- [Encoder] 正在预热 mlpackage (固定形状: {self.dml_pad_to}s)... ---")
            dummy_wav = np.zeros(int(16000 * self.dml_pad_to)).astype(np.float32)
            _ = self.encode(dummy_wav)
        else:
            # 非 DML 模式下，预热用 dml_pad_to 大小 (>= chunk_size_sec) 让 numpy.fft / BLAS
            # 在冷启动时就按真实 chunk size 构建 plan 与 thread pool, 避免 chunk 1 第一次
            # 跑 40s 真实 mel 时 fft plan miss / BLAS spawn 触发 ~10s cold start.
            # 必须用 non-zero data: np.zeros 在 BLAS/MKL 的 dot product 可能走 sparse
            # shortcut, 不会触发完整 GEMM kernel JIT, warmup 失效.
            warmup_sec = max(self.dml_pad_to, 2.0)
            if self.verbose: print(f"--- [Encoder] 正在预热 (非 DML 模式, {warmup_sec}s)... ---")
            _rng = np.random.default_rng(0)
            dummy_wav = _rng.standard_normal(int(16000 * warmup_sec)).astype(np.float32) * 0.1
            _ = self.encode(dummy_wav)
        if self.verbose: print("--- [Encoder] 预热完成 ---")

    def _run_frontend(self, mel: np.ndarray) -> np.ndarray:
        """前端推理流水线：Pad -> Chunk Loop -> Concat -> Slice"""
        T = mel.shape[1]
        
        # 1. 必须 Pad 到 100 的倍数
        pad_len = (100 - (T % 100)) % 100
        if pad_len > 0:
            mel = np.pad(mel, ((0,0), (0, pad_len)), mode='constant')
        
        # 增加 batch 维 -> (1, 128, T_padded)
        mel_input = mel[np.newaxis, ...]
        
        num_chunks = mel_input.shape[2] // 100
        fe_outputs = []
        chunk_size = 100
        
        # 2. 循环推理 (Atomic Inference)
        for i in range(num_chunks):
            start = i * chunk_size
            chunk = mel_input[:, :, start : start + chunk_size]
            out = self.sess_fe.run(None, {"chunk_mel": chunk})[0] # (1, 13, 896/1024)
            fe_outputs.append(out)
            
        # 3. 拼接结果 -> (1, N_frames, D)
        hidden_states = np.concatenate(fe_outputs, axis=1)
        
        # 4. 有效长度切片 (关键: 去除 Padding 带来的尾部垃圾帧)
        t_out = get_feat_extract_output_lengths(T)
        hidden_states = hidden_states[:, :t_out, :]
        
        return hidden_states

    def _run_backend(self, hidden_states: np.ndarray) -> np.ndarray:
        """后端推理流水线：Mask -> Transformer (支持固定形状 Padding + mlpackage)"""
        batch, seq_len, dim = hidden_states.shape

        # 路径 A: COREML_ANE_FULL mlpackage backend (static shape (1, h_target_len, dim))
        if self.sess_be_mlmodel is not None:
            target_t = self.h_target_len
            if seq_len > target_t:
                # 超过 static shape 不支持; 截断 + warn (上层 chunk_size_sec 应该匹配 dml_pad_to)
                if self.verbose:
                    print(f"--- [Encoder] WARN: mlpackage static T={target_t}, 输入 seq_len={seq_len}, 截断 ---")
                hidden_states = hidden_states[:, :target_t, :]
                seq_len = target_t
            if seq_len < target_t:
                pad_width = target_t - seq_len
                hidden_input = np.pad(hidden_states, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
            else:
                hidden_input = hidden_states
            # 2D key_padding_mask: 前 seq_len = 1 (有效), 后 = 0 (pad)
            key_padding_mask = np.zeros((batch, target_t), dtype=np.int32)
            key_padding_mask[:, :seq_len] = 1
            # mlpackage 接口是 fp32 (compute_precision=FLOAT16 内部 cast)
            pred = self.sess_be_mlmodel.predict({
                "hidden_states": hidden_input.astype(np.float32),
                "key_padding_mask": key_padding_mask,
            })
            audio_embd = pred["last_hidden_state"]
            # 切回有效长度
            if audio_embd.shape[1] > seq_len:
                audio_embd = audio_embd[:, :seq_len, :]
            return audio_embd

        # 路径 B: ONNX backend (CPU / GPU 等), 跟 Phase 2 行为一致
        # 1. 形状检查与 Padding (仅在 DML 开启时执行)
        if self.active_dml and seq_len < self.h_target_len:
            pad_width = self.h_target_len - seq_len
            # 对 hidden_states 进行零填充 -> (Batch, T_fixed, D)
            hidden_input = np.pad(hidden_states, ((0, 0), (0, pad_width), (0, 0)), mode='constant')

            # 构造 Mask：前 seq_len 为 0 (关注)，后 pad_width 为 -10000.0 (屏蔽)
            # 维度需要广播到 (Batch, 1, T_fixed, T_fixed)
            mask = np.zeros((batch, 1, self.h_target_len, self.h_target_len), dtype=self.input_dtype)
            mask[:, :, :, seq_len:] = -10000.0
        else:
            hidden_input = hidden_states
            mask = np.zeros((batch, 1, seq_len, seq_len), dtype=self.input_dtype)

        # 2. 执行推理
        audio_embd = self.sess_be.run(None, {
            "hidden_states": hidden_input,
            "attention_mask": mask
        })[0]

        # 3. 截断输出 -> (Batch, seq_len, D)
        if audio_embd.shape[1] > seq_len:
            audio_embd = audio_embd[:, :seq_len, :]

        return audio_embd

    def encode(self, audio: np.ndarray) -> tuple:
        """执行编码 (Mel -> Frontend -> Backend)，返回 (embedding, 耗时)"""
        t0 = time.time()

        # 1. 提取 Mel 特征
        # audio: (N_samples,) -> mel: (128, T)
        t_mel = time.time()
        mel = self.mel_extractor(audio, dtype=self.input_dtype)
        dt_mel = time.time() - t_mel

        # 2. Frontend (Loop)
        t_fe = time.time()
        hidden_states = self._run_frontend(mel)
        dt_fe = time.time() - t_fe

        # 3. Backend (Transformer)
        t_be = time.time()
        audio_embd = self._run_backend(hidden_states)
        dt_be = time.time() - t_be

        # 4. 去除 Batch 维 -> (T, D)
        if audio_embd.ndim == 3:
            audio_embd = audio_embd[0]

        elapsed = time.time() - t0
        if os.environ.get("FUNASR_QWEN3_ENCODER_TIMING") == "1":
            print(
                f"[encoder-timing] mel={dt_mel*1000:.1f}ms "
                f"frontend={dt_fe*1000:.1f}ms backend={dt_be*1000:.1f}ms "
                f"total={elapsed*1000:.1f}ms",
                flush=True,
            )
        return audio_embd, elapsed
