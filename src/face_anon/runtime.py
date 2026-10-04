from __future__ import annotations

import onnxruntime as ort

_preloaded = False


def _preload_cuda() -> None:
    # CUDA / cuDNN come from the pip nvidia-* wheels (shipped with torch), not
    # the system path, so ORT has to be told to load them first.
    global _preloaded
    if _preloaded:
        return
    _preloaded = True
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(cuda=True, cudnn=True, msvc=False)
    except Exception:  # noqa: BLE001
        pass


def onnx_providers(device: str = "auto") -> list[str]:
    """`auto` uses CUDA when onnxruntime-gpu can see it, else CPU."""
    device = (device or "auto").lower()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        _preload_cuda()
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        raise RuntimeError("device=cuda requested but onnxruntime-gpu has no CUDAExecutionProvider")
    return ["CPUExecutionProvider"]


def make_session(path: str, device: str = "auto", threads: int = 4) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=so, providers=onnx_providers(device))


def session_device(session: ort.InferenceSession) -> str:
    return "cuda" if session.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
