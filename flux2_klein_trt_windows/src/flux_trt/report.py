from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .diagnostics import local_timestamp_for_path, utc_timestamp, write_json


def create_run_dir(output_root: str | Path, prefix: str = "run") -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / f"{prefix}_{local_timestamp_for_path()}"
    index = 1
    while candidate.exists():
        candidate = root / f"{prefix}_{local_timestamp_for_path()}_{index}"
        index += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def base_run_report(
    *,
    variant: str,
    apacheone_checkpoint: str,
    base_repo: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    mode: str = "cached_embeddings_strict",
) -> dict[str, Any]:
    return {
        "mode": mode,
        "variant": variant,
        "apacheone_checkpoint": apacheone_checkpoint,
        "base_repo": base_repo,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "device": "cuda",
        "gpu_name": None,
        "cuda_capability": None,
        "is_blackwell_or_newer": None,
        "nvfp4_target_gpu": None,
        "vram_total_gb": None,
        "vram_free_before_load": None,
        "vram_peak_allocated_gb": None,
        "docker_image": None,
        "torch_version": None,
        "tensorrt_llm_version": None,
        "model_dir": None,
        "worker_stdout_path": None,
        "worker_stderr_path": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "detected_oom": False,
        "detected_unsupported_arch": False,
        "detected_missing_model_index": False,
        "detected_invalid_safetensors": False,
        "traceback": None,
        "load_time_sec": None,
        "generation_time_sec": None,
        "total_time_sec": None,
        "prompt_cache_used": mode == "cached_embeddings_strict",
        "user_photo_cache_used": True,
        "logo_cache_used": True,
        "smoke_test_only": mode == "visualgen_prompt_text",
        "status": "pending",
        "error": None,
        "created_at": utc_timestamp(),
    }


def write_run_report(output_dir: str | Path, report: dict[str, Any]) -> Path:
    return write_json(Path(output_dir) / "run_report.json", report)


def copy_if_exists(source: str | Path, target: str | Path) -> bool:
    source_path = Path(source)
    if not source_path.exists():
        return False
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return True
