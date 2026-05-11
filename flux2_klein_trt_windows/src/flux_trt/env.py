from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUNTIME_IMPORTS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("huggingface_hub", "huggingface_hub"),
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("yaml", "pyyaml"),
    ("pydantic", "pydantic"),
    ("psutil", "psutil"),
    ("rich", "rich"),
    ("tqdm", "tqdm"),
    ("tensorrt", "tensorrt"),
    ("tensorrt_llm", "tensorrt-llm"),
]

DEFAULT_DOCKER_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14"


@dataclass(frozen=True)
class ImportStatus:
    module: str
    package: str
    available: bool
    version: str | None
    error: str | None


def probe_import(module_name: str, package_name: str | None = None) -> ImportStatus:
    package = package_name or module_name
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        if version is None:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = None
        return ImportStatus(module_name, package, True, version, None)
    except Exception as exc:  # noqa: BLE001 - diagnostics should capture import failures.
        return ImportStatus(module_name, package, False, None, repr(exc))


def _run_nvidia_smi() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False, "error": "nvidia-smi not found in PATH"}

    try:
        completed = subprocess.run(
            [
                exe,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}

    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append(
                {
                    "name": parts[0],
                    "memory_total_mb": _safe_int(parts[1]),
                    "driver_version": parts[2],
                }
            )
    return {"available": True, "gpus": rows}


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _torch_cuda_info() -> dict[str, Any]:
    status = probe_import("torch")
    if not status.available:
        return {"torch_available": False, "error": status.error}

    import torch

    info: dict[str, Any] = {
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": None,
        "vram_total_gb": None,
        "vram_free_before_load_gb": None,
        "vram_free_before_load_bytes": None,
        "cuda_capability": None,
        "is_blackwell_or_newer": False,
        "nvfp4_target_gpu": False,
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        cuda_capability = list(torch.cuda.get_device_capability(device_index))
        is_blackwell_or_newer = _is_blackwell_or_newer(cuda_capability)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        info.update(
            {
                "device_index": device_index,
                "gpu_name": props.name,
                "vram_total_gb": round(total_bytes / (1024**3), 2),
                "vram_free_before_load_gb": round(free_bytes / (1024**3), 3),
                "vram_free_before_load_bytes": int(free_bytes),
                "multi_processor_count": props.multi_processor_count,
                "cuda_capability": cuda_capability,
                "is_blackwell_or_newer": is_blackwell_or_newer,
                "nvfp4_target_gpu": is_blackwell_or_newer,
            }
        )
    return info


def _is_blackwell_or_newer(cuda_capability: list[int] | tuple[int, int] | None) -> bool:
    if not cuda_capability:
        return False
    try:
        major = int(cuda_capability[0])
    except (TypeError, ValueError):
        return False
    return major >= 12


def running_in_container() -> bool:
    return Path("/.dockerenv").exists() or bool(os.environ.get("container"))


def _allow_container_from_env() -> bool:
    return os.environ.get("FLUX_ALLOW_DOCKER", "").strip().lower() in {"1", "true", "yes"}


def collect_env_info(*, allow_container: bool = False) -> dict[str, Any]:
    allow_container = allow_container or _allow_container_from_env()
    imports = [probe_import(module, package).__dict__ for module, package in RUNTIME_IMPORTS]
    import_map = {item["module"]: item for item in imports}
    errors = []
    warnings = []

    system = platform.system().lower()
    in_container = running_in_container()
    if system != "windows":
        if allow_container and system == "linux" and in_container:
            warnings.append(
                "Running inside a Linux Docker container. This is accepted because Docker mode is enabled."
            )
        else:
            errors.append("This project target is Windows 11 x64 or the approved Docker container runtime.")
    if system == "windows" and allow_container:
        warnings.append("Docker mode is enabled, but env_check is running on the Windows host.")
    if sys.version_info[:2] != (3, 12):
        warnings.append(
            f"Python 3.12 is preferred; current version is {platform.python_version()}."
        )

    for module_name in ["tensorrt", "tensorrt_llm"]:
        if not import_map[module_name]["available"]:
            errors.append(f"Missing runtime dependency: {module_name}")

    torch_cuda = _torch_cuda_info()
    if not torch_cuda.get("cuda_available"):
        errors.append("torch.cuda.is_available() is false.")

    cuda_capability = torch_cuda.get("cuda_capability")
    is_blackwell_or_newer = bool(torch_cuda.get("is_blackwell_or_newer", False))
    nvfp4_target_gpu = bool(torch_cuda.get("nvfp4_target_gpu", False))
    docker_image = os.environ.get("FLUX_DOCKER_IMAGE") or os.environ.get(
        "NVIDIA_CONTAINER_IMAGE", DEFAULT_DOCKER_IMAGE
    )

    status = "ok" if not errors else "error"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "cuda_capability": cuda_capability,
        "is_blackwell_or_newer": is_blackwell_or_newer,
        "nvfp4_target_gpu": nvfp4_target_gpu,
        "gpu_name": torch_cuda.get("gpu_name"),
        "vram_total_gb": torch_cuda.get("vram_total_gb"),
        "vram_free_before_load": torch_cuda.get("vram_free_before_load_gb"),
        "docker_image": docker_image,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "running_in_container": in_container,
        "allow_container": allow_container,
        "imports": imports,
        "torch_cuda": torch_cuda,
        "nvidia_smi": _run_nvidia_smi(),
    }


def require_generation_environment(*, allow_container: bool = False) -> dict[str, Any]:
    info = collect_env_info(allow_container=allow_container)
    if info["status"] != "ok":
        joined = "; ".join(info["errors"])
        raise RuntimeError(f"Environment is not ready for generation: {joined}")
    return info
