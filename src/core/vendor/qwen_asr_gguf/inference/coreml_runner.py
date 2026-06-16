# coding=utf-8
"""
PyObjC zero-copy CoreML runner.

替代 coremltools.models.MLModel.predict() 路径,规避 macOS 26 + coremltools 9.0
的 MLE5ExecutionStream lingering reset SIGSEGV race (Gemini deep research C.6 验证).

核心机制:
- MLMultiArray.initWithDataPointer + deallocator=None: ObjC 只持 numpy 裸 C 指针,
  不 retain Python 对象
- MLE5 后台 dispatch worker 释放 MLMultiArray 时, 不调 _PyObject_Free, 不踩 GIL race
- numpy buffer 由 Python 主线程 gc 管理 (有 GIL 时), 安全

PoC 验证: 20 predict + sherpa-onnx 共存 + 120s lingering 等 = 无 SIGSEGV.
详见 spikes/qwen3_mac_hw_accel/phase3_backend/poc_pyobjc_zerocopy.py
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# CoreML constants (MLMultiArray.h)
_MLMultiArrayDataTypeFloat32 = 0x10020
_MLMultiArrayDataTypeInt32 = 0x20020
_MLMultiArrayDataTypeFloat16 = 0x10010
# MLComputeUnits enum
_MLComputeUnitsCPUOnly = 0
_MLComputeUnitsCPUAndGPU = 1
_MLComputeUnitsCPUAndNeuralEngine = 2
_MLComputeUnitsAll = 3

_NP_DTYPE_TO_ML = {
    np.dtype(np.float32): _MLMultiArrayDataTypeFloat32,
    np.dtype(np.int32): _MLMultiArrayDataTypeInt32,
    np.dtype(np.float16): _MLMultiArrayDataTypeFloat16,
}
_ML_DTYPE_TO_NP = {v: k for k, v in _NP_DTYPE_TO_ML.items()}


def _lazy_load_coreml():
    """Lazy load PyObjC + CoreML framework. Returns (module-level cls handles)."""
    import objc
    objc.loadBundle(
        "CoreML",
        bundle_path="/System/Library/Frameworks/CoreML.framework",
        module_globals=globals(),
    )
    return {
        "objc": objc,
        "MLModel": objc.lookUpClass("MLModel"),
        "MLModelConfiguration": objc.lookUpClass("MLModelConfiguration"),
        "MLMultiArray": objc.lookUpClass("MLMultiArray"),
        "MLDictionaryFeatureProvider": objc.lookUpClass("MLDictionaryFeatureProvider"),
    }


class CoreMLZeroCopyRunner:
    """PyObjC zero-copy MLModel runner.

    用法:
        runner = CoreMLZeroCopyRunner(mlpackage_path, compute_units="CPU_AND_NE")
        out = runner.predict({"hidden_states": np_h, "key_padding_mask": np_m})
        # out 是 dict {output_name: np.ndarray (copy)}
    """

    _COMPUTE_UNITS = {
        "CPU_ONLY": _MLComputeUnitsCPUOnly,
        "CPU_AND_GPU": _MLComputeUnitsCPUAndGPU,
        "CPU_AND_NE": _MLComputeUnitsCPUAndNeuralEngine,
        "ALL": _MLComputeUnitsAll,
    }

    def __init__(self, mlpackage_path: str, compute_units: str = "CPU_AND_NE", verbose: bool = True):
        self.verbose = verbose
        self._classes = _lazy_load_coreml()
        from Foundation import NSURL
        self._NSURL = NSURL

        import os
        p = os.path.abspath(mlpackage_path)
        url = NSURL.fileURLWithPath_(p)

        if verbose:
            print(f"--- [CoreMLRunner] compileModelAtURL: {p}", flush=True)
        compiled_url = self._classes["MLModel"].compileModelAtURL_error_(url, None)
        if compiled_url is None:
            raise RuntimeError(f"CoreML compileModelAtURL failed for {p}")

        config = self._classes["MLModelConfiguration"].alloc().init()
        cu = self._COMPUTE_UNITS.get(compute_units.upper())
        if cu is None:
            raise ValueError(f"Unknown compute_units={compute_units!r}, expected one of {list(self._COMPUTE_UNITS)}")
        config.setComputeUnits_(cu)

        if verbose:
            print(f"--- [CoreMLRunner] modelWithContentsOfURL + {compute_units} (cold start: ANE plan compile ~30s)", flush=True)
        self._mlmodel = self._classes["MLModel"].modelWithContentsOfURL_configuration_error_(
            compiled_url, config, None,
        )
        if self._mlmodel is None:
            raise RuntimeError(f"CoreML modelWithContentsOfURL failed for {compiled_url}")

        # 缓存 input/output 元数据 (用于校验和类型转换)
        desc = self._mlmodel.modelDescription()
        self._input_names = list(desc.inputDescriptionsByName().keys())
        self._output_names = list(desc.outputDescriptionsByName().keys())
        if verbose:
            print(f"--- [CoreMLRunner] ready: inputs={self._input_names}, outputs={self._output_names}", flush=True)

    @property
    def input_names(self):
        return list(self._input_names)

    @property
    def output_names(self):
        return list(self._output_names)

    def _np_to_mlarray(self, arr: np.ndarray):
        """Zero-copy numpy → MLMultiArray (initWithDataPointer + deallocator=None).

        - deallocator=None: MLMultiArray 不释放 buffer, numpy 自己管
        - 调用方必须保 numpy ref 在 predict 调用期间存活
        """
        from Foundation import NSNumber
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        mltype = _NP_DTYPE_TO_ML.get(arr.dtype)
        if mltype is None:
            raise ValueError(f"Unsupported numpy dtype {arr.dtype}; supported: {list(_NP_DTYPE_TO_ML)}")
        shape = [NSNumber.numberWithLong_(s) for s in arr.shape]
        # strides 单位: numpy=bytes, MLMultiArray=elements
        strides = [NSNumber.numberWithLong_(s // arr.itemsize) for s in arr.strides]
        ml = self._classes["MLMultiArray"].alloc().initWithDataPointer_shape_dataType_strides_deallocator_error_(
            arr.ctypes.data, shape, mltype, strides, None, None,
        )
        if ml is None:
            raise RuntimeError("MLMultiArray.initWithDataPointer failed")
        return ml

    def _mlarray_to_np(self, ml_array) -> np.ndarray:
        """MLMultiArray → numpy. Copy out to detach from ObjC lifecycle."""
        shape_obj = ml_array.shape()
        shape = tuple(int(shape_obj[k]) for k in range(shape_obj.count()))
        dt = int(ml_array.dataType())
        np_dtype = _ML_DTYPE_TO_NP.get(dt)
        if np_dtype is None:
            raise RuntimeError(f"Unsupported MLMultiArrayDataType {dt}")
        # 取裸指针 + 拷贝
        ptr = ml_array.dataPointer()
        # ml_array.count() 返回元素数
        count = int(ml_array.count())
        # 用 numpy.frombuffer 从指针读 (zero-copy view) 然后 copy
        try:
            import ctypes
            buf_type = ctypes.c_byte * (count * np_dtype.itemsize)
            buf = buf_type.from_address(int(ptr))
            view = np.frombuffer(buf, dtype=np_dtype).reshape(shape)
            return view.copy()
        except Exception:
            # fallback: 用 ObjC getBytes (慢但稳)
            arr = np.empty(shape, dtype=np_dtype)
            buf_type = ctypes.c_byte * arr.nbytes
            ml_array.getBytes_length_(buf_type.from_address(arr.ctypes.data), arr.nbytes)
            return arr

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference. Returns {output_name: np.ndarray (owned copy)}."""
        # 1. numpy → MLMultiArray (zero-copy). 保 numpy ref 到 return 前不丢
        ml_inputs = {}
        for name, arr in inputs.items():
            # 持 numpy ref 在 ml_inputs 一边, 保证不 gc
            ml_inputs[name] = (arr, self._np_to_mlarray(arr))

        # 2. 构造 FeatureProvider
        feat_dict = {name: ml for name, (_, ml) in ml_inputs.items()}
        provider = self._classes["MLDictionaryFeatureProvider"].alloc().initWithDictionary_error_(feat_dict, None)
        if provider is None:
            raise RuntimeError("MLDictionaryFeatureProvider init failed")

        # 3. predict
        pred = self._mlmodel.predictionFromFeatures_error_(provider, None)
        if pred is None:
            raise RuntimeError("MLModel.predictionFromFeatures failed")

        # 4. 收集 outputs (copy out 脱离 ObjC 生命周期)
        out = {}
        for name in self._output_names:
            mla = pred.featureValueForName_(name).multiArrayValue()
            out[name] = self._mlarray_to_np(mla)

        # 5. ml_inputs 持的 numpy ref 在这里 release; ml_inputs 字典退出函数后 gc
        return out
