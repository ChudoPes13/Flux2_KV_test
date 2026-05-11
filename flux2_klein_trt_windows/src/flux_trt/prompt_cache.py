from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Mapping

from .diagnostics import utc_timestamp, write_json
from .hashing import sha256_text
from .tensor_io import load_tensors_safetensors, save_tensors_safetensors, tensor_summary


def build_prompt_metadata(
    *,
    prompt_id: str,
    prompt: str,
    dtype: str,
    text_encoder_source: str,
    tokenizer_source: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "prompt_id": prompt_id,
        "prompt_sha256": sha256_text(prompt),
        "text_encoder_source": text_encoder_source,
        "tokenizer_source": tokenizer_source,
        "dtype": dtype,
        "created_at": utc_timestamp(),
        "python_version": platform.python_version(),
        "torch_version": _version("torch"),
        "transformers_version": _version("transformers"),
        "diffusers_version": _version("diffusers"),
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def save_prompt_cache(
    *,
    cache_dir: str | Path,
    prompt: str,
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    output_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    tensor_path = save_tensors_safetensors(tensors, output_dir / "prompt_tensors.safetensors")
    enriched_metadata = dict(metadata)
    enriched_metadata["tensor_file"] = str(tensor_path)
    enriched_metadata["tensors"] = tensor_summary(tensors)
    meta_path = write_json(output_dir / "prompt_meta.json", enriched_metadata)

    return {
        "prompt": prompt_path,
        "metadata": meta_path,
        "tensors": tensor_path,
    }


def load_prompt_cache(cache_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = Path(cache_dir)
    tensor_path = directory / "prompt_tensors.safetensors"
    meta_path = directory / "prompt_meta.json"
    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing prompt tensor cache: {tensor_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing prompt metadata: {meta_path}")

    tensors = load_tensors_safetensors(tensor_path)
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return tensors, metadata

