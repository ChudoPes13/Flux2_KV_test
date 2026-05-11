from __future__ import annotations

import json
import platform
import traceback as traceback_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .env import collect_env_info


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_timestamp_for_path() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output_path


def write_diagnostic(
    diagnostics_dir: str | Path,
    stage: str,
    message: str,
    *,
    status: str = "error",
    exc: BaseException | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    diagnostics_path = Path(diagnostics_dir)
    diagnostics_path.mkdir(parents=True, exist_ok=True)

    try:
        env = collect_env_info()
    except Exception as env_exc:  # noqa: BLE001
        env = {"status": "error", "message": f"Could not collect env info: {env_exc!r}"}

    torch_cuda = env.get("torch_cuda", {}) if isinstance(env, dict) else {}
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "message": message,
        "traceback": "".join(traceback_module.format_exception(exc)) if exc else None,
        "created_at": utc_timestamp(),
        "python_version": platform.python_version(),
        "torch_version": torch_cuda.get("torch_version"),
        "cuda_available": torch_cuda.get("cuda_available"),
        "gpu_name": torch_cuda.get("gpu_name"),
        "vram_total_gb": torch_cuda.get("vram_total_gb"),
        "env": env,
    }
    if extra:
        payload.update(extra)

    base = diagnostics_path / f"diagnostic_{local_timestamp_for_path()}.json"
    path = base
    index = 1
    while path.exists():
        path = diagnostics_path / f"{base.stem}_{index}.json"
        index += 1
    return write_json(path, payload)
