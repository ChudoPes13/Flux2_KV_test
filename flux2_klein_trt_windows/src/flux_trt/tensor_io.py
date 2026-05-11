from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


COMMON_ENCODE_PROMPT_NAMES = [
    "prompt_embeds",
    "pooled_prompt_embeds",
    "text_ids",
    "attention_mask",
    "negative_prompt_embeds",
    "negative_pooled_prompt_embeds",
]


def _is_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


def _clean_key(key: str) -> str:
    return (
        key.replace(" ", "_")
        .replace("/", ".")
        .replace("\\", ".")
        .replace("[", "_")
        .replace("]", "")
    )


def flatten_tensors(value: Any, prefix: str = "tensor") -> dict[str, Any]:
    tensors: dict[str, Any] = {}

    def walk(item: Any, name: str) -> None:
        if _is_tensor(item):
            tensors[_clean_key(name)] = item
            return
        if isinstance(item, Mapping):
            for key, sub_item in item.items():
                walk(sub_item, f"{name}.{key}")
            return
        if isinstance(item, tuple):
            for index, sub_item in enumerate(item):
                common = COMMON_ENCODE_PROMPT_NAMES[index] if index < len(COMMON_ENCODE_PROMPT_NAMES) else str(index)
                walk(sub_item, common)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, sub_item in enumerate(item):
                walk(sub_item, f"{name}.{index}")

    walk(value, prefix)
    return tensors


def tensor_summary(tensors: Mapping[str, Any]) -> dict[str, Any]:
    summary = {}
    for name, tensor in tensors.items():
        dtype = str(getattr(tensor, "dtype", "unknown"))
        shape = list(getattr(tensor, "shape", []))
        device = str(getattr(tensor, "device", "unknown"))
        summary[name] = {"shape": shape, "dtype": dtype, "device": device}
    return summary


def save_tensors_safetensors(tensors: Mapping[str, Any], path: str | Path) -> Path:
    if not tensors:
        raise ValueError("No tensors were found to save.")
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("Missing dependency: safetensors.") from exc

    clean = {}
    for name, tensor in tensors.items():
        if not _is_tensor(tensor):
            continue
        clean[_clean_key(name)] = tensor.detach().cpu().contiguous()
    if not clean:
        raise ValueError("No torch tensors were found to save.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(clean, str(output_path))
    return output_path


def load_tensors_safetensors(path: str | Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("Missing dependency: safetensors.") from exc
    return load_file(str(path), device="cpu")

