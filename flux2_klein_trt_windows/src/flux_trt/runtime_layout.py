from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .config import ProjectConfig


RUNTIME_ROOT = Path("data/cache/visualgen_runtime")


def runtime_dir_for_variant(config: ProjectConfig, variant: str) -> Path:
    return config.resolve_path(RUNTIME_ROOT / variant)


def prepare_visualgen_runtime_dir(config: ProjectConfig, variant: str) -> dict[str, Any]:
    runtime_dir = runtime_dir_for_variant(config, variant)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    bfl_dir = config.model_dir("bfl_dir")
    text_encoder_dir = config.resolve_path(
        config.raw["experimental"]["text_encoder"]["variants"]["aifeifei_4bit"]["local_dir"]
    )
    checkpoint_path = config.checkpoint_path(variant)

    operations = []
    required = [
        bfl_dir / "model_index.json",
        bfl_dir / "scheduler" / "scheduler_config.json",
        bfl_dir / "tokenizer" / "tokenizer_config.json",
        bfl_dir / "tokenizer" / "tokenizer.json",
        bfl_dir / "vae" / "config.json",
        bfl_dir / "vae" / "diffusion_pytorch_model.safetensors",
        bfl_dir / "transformer" / "config.json",
        text_encoder_dir / "config.json",
        text_encoder_dir / "model.safetensors",
        checkpoint_path,
    ]
    missing = [str(path) for path in required if not path.exists()]

    if not missing:
        operations.append(_copy_file(bfl_dir / "model_index.json", runtime_dir / "model_index.json"))
        operations.append(_link_or_copy_dir(bfl_dir / "scheduler", runtime_dir / "scheduler"))
        operations.append(_link_or_copy_dir(bfl_dir / "tokenizer", runtime_dir / "tokenizer"))
        operations.append(_link_or_copy_dir(bfl_dir / "vae", runtime_dir / "vae"))
        operations.append(_link_or_copy_dir(text_encoder_dir, runtime_dir / "text_encoder"))

        transformer_dir = runtime_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=True)
        operations.append(
            _copy_file(bfl_dir / "transformer" / "config.json", transformer_dir / "config.json")
        )
        operations.append(
            _link_or_copy_file(
                checkpoint_path,
                transformer_dir / "diffusion_pytorch_model.safetensors",
            )
        )

    return {
        "variant": variant,
        "runtime_dir": str(runtime_dir),
        "checkpoint_path": str(checkpoint_path),
        "bfl_dir": str(bfl_dir),
        "text_encoder_dir": str(text_encoder_dir),
        "missing": missing,
        "operations": operations,
        "layout": inspect_runtime_layout(runtime_dir),
    }


def inspect_runtime_layout(runtime_dir: str | Path) -> dict[str, Any]:
    root = Path(runtime_dir)
    expected = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    return {
        "root": str(root),
        "exists": _safe_exists(root),
        "expected": [_inspect_path(root / item) for item in expected],
    }


def _copy_file(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if _lexists(target):
        _remove_existing_path(target)
    shutil.copy2(source, target)
    return {"op": "copy_file", "source": str(source), "target": str(target)}


def _link_or_copy_file(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if _lexists(target):
        _remove_existing_path(target)
    try:
        os.symlink(source, target)
        return {"op": "symlink_file", "source": str(source), "target": str(target)}
    except OSError:
        try:
            os.link(source, target)
            return {"op": "hardlink_file", "source": str(source), "target": str(target)}
        except OSError:
            shutil.copy2(source, target)
            return {"op": "copy_file", "source": str(source), "target": str(target)}


def _link_or_copy_dir(source: Path, target: Path) -> dict[str, Any]:
    if _lexists(target):
        _remove_existing_path(target)
    try:
        os.symlink(source, target, target_is_directory=True)
        return {"op": "symlink_dir", "source": str(source), "target": str(target)}
    except OSError:
        copied = []
        hardlinked = []
        target.mkdir(parents=True, exist_ok=True)
        for item in source.rglob("*"):
            if ".cache" in item.parts:
                continue
            rel = item.relative_to(source)
            dest = target / rel
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(item, dest)
                hardlinked.append(str(rel))
            except OSError:
                shutil.copy2(item, dest)
                copied.append(str(rel))
        return {
            "op": "hardlink_or_copy_dir",
            "source": str(source),
            "target": str(target),
            "hardlinked": hardlinked,
            "copied": copied,
        }


def _inspect_path(path: Path) -> dict[str, Any]:
    exists = _safe_exists(path)
    return {
        "path": str(path),
        "exists": exists,
        "is_symlink": _safe_is_symlink(path),
        "size_bytes": _safe_size(path) if exists else None,
        "inaccessible": _lexists(path) and not exists,
    }


def _lexists(path: Path) -> bool:
    try:
        return os.path.lexists(path)
    except OSError:
        return False


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _safe_is_symlink(path: Path) -> bool:
    try:
        return path.is_symlink()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _remove_existing_path(path: Path) -> None:
    if _safe_is_symlink(path) or _safe_is_file(path):
        path.unlink()
        return
    try:
        shutil.rmtree(path)
        return
    except (NotADirectoryError, OSError):
        pass
    try:
        path.rmdir()
        return
    except OSError:
        pass
    path.unlink()
