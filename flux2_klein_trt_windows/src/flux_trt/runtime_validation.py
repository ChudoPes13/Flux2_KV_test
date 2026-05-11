from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .runtime_layout import prepare_visualgen_runtime_dir


SOURCE_CHECKS = [
    ("model_index", "models/bfl/model_index.json"),
    ("scheduler_config", "models/bfl/scheduler/scheduler_config.json"),
    ("tokenizer_config", "models/bfl/tokenizer/tokenizer_config.json"),
    ("tokenizer_json", "models/bfl/tokenizer/tokenizer.json"),
    ("vae_config", "models/bfl/vae/config.json"),
    ("vae_weights", "models/bfl/vae/diffusion_pytorch_model.safetensors"),
    ("transformer_config", "models/bfl/transformer/config.json"),
]


def validate_runtime_variant(config: ProjectConfig, variant: str) -> dict[str, Any]:
    bfl_dir = config.model_dir("bfl_dir")
    source_paths = {
        "model_index": bfl_dir / "model_index.json",
        "scheduler_config": bfl_dir / "scheduler" / "scheduler_config.json",
        "tokenizer_config": bfl_dir / "tokenizer" / "tokenizer_config.json",
        "tokenizer_json": bfl_dir / "tokenizer" / "tokenizer.json",
        "vae_config": bfl_dir / "vae" / "config.json",
        "vae_weights": bfl_dir / "vae" / "diffusion_pytorch_model.safetensors",
        "transformer_config": bfl_dir / "transformer" / "config.json",
        "checkpoint": config.checkpoint_path(variant),
    }

    source_checks = [
        _path_check(name, path, parse_json=name.endswith("_config") or name == "model_index")
        for name, path in source_paths.items()
    ]
    layout = prepare_visualgen_runtime_dir(config, variant)
    runtime_checks = [
        _path_check(item["path"], Path(item["path"]))
        for item in layout["layout"]["expected"]
    ]

    errors = []
    errors.extend(f"missing source {item['name']}: {item['path']}" for item in source_checks if not item["exists"])
    errors.extend(f"invalid json {item['name']}: {item['json_error']}" for item in source_checks if item.get("json_error"))
    errors.extend(f"missing runtime file: {item['path']}" for item in runtime_checks if not item["exists"])
    errors.extend(f"runtime layout missing: {item}" for item in layout.get("missing", []))

    return {
        "variant": variant,
        "status": "ok" if not errors else "error",
        "errors": errors,
        "source_checks": source_checks,
        "runtime_layout": layout,
        "runtime_checks": runtime_checks,
        "gpu_loaded": False,
    }


def _path_check(name: str, path: Path, *, parse_json: bool = False) -> dict[str, Any]:
    exists = _safe_exists(path)
    result: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "exists": exists,
        "is_file": _safe_is_file(path) if exists else False,
        "is_dir": _safe_is_dir(path) if exists else False,
        "is_symlink": _safe_is_symlink(path),
        "size_bytes": _safe_size(path) if exists and _safe_is_file(path) else None,
        "inaccessible": _safe_is_symlink(path) and not exists,
    }
    if parse_json and exists:
        try:
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            result["json_ok"] = True
            result["json_error"] = None
        except Exception as exc:  # noqa: BLE001
            result["json_ok"] = False
            result["json_error"] = repr(exc)
    return result


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_symlink(path: Path) -> bool:
    try:
        return path.is_symlink()
    except OSError:
        return False


def _safe_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None
