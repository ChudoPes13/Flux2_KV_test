from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_PROMPT_EMBEDS_SHAPE = [1, 512, 12288]

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing safetensors checkpoint: {checkpoint}")
    with checkpoint.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Invalid safetensors file, missing header size: {checkpoint}")
        header_size = struct.unpack("<Q", size_bytes)[0]
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"Invalid safetensors file, incomplete header: {checkpoint}")
    return json.loads(header_bytes.decode("utf-8"))


def inspect_safetensors_checkpoint(
    path: str | Path,
    *,
    expected_prompt_embeds_shape: list[int] | tuple[int, ...] = EXPECTED_PROMPT_EMBEDS_SHAPE,
) -> dict[str, Any]:
    checkpoint = Path(path)
    header = read_safetensors_header(checkpoint)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    dtype_counts = Counter(str(info.get("dtype", "unknown")) for info in tensors.values())
    prefix_counts = Counter(_key_prefix(key) for key in tensors)
    tensor_shapes = {
        key: {
            "dtype": str(info.get("dtype", "unknown")),
            "shape": list(info.get("shape", [])),
            "numel": _numel(info.get("shape", [])),
            "size_bytes": _tensor_size_bytes(info),
        }
        for key, info in tensors.items()
    }

    key_shape_samples = {
        key: tensor_shapes[key]
        for key in sorted(tensor_shapes)[:50]
    }
    key_shape_samples.update(
        {
            key: tensor_shapes[key]
            for key in _interesting_keys(tensor_shapes)
            if key in tensor_shapes
        }
    )

    txt_in_key = _find_key(tensor_shapes, "txt_in.weight")
    txt_in_shape = tensor_shapes[txt_in_key]["shape"] if txt_in_key else None
    expected_hidden = list(expected_prompt_embeds_shape)[-1]
    txt_in_compatible = (
        txt_in_shape is not None
        and len(txt_in_shape) >= 2
        and int(txt_in_shape[-1]) == int(expected_hidden)
    )

    return {
        "checkpoint_path": str(checkpoint),
        "exists": checkpoint.exists(),
        "file_size_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
        "tensor_count": len(tensors),
        "metadata": header.get("__metadata__", {}),
        "dtypes": dict(sorted(dtype_counts.items())),
        "key_prefixes": [
            {"prefix": prefix, "count": count}
            for prefix, count in prefix_counts.most_common()
        ],
        "key_shape_samples": key_shape_samples,
        "largest_tensors": _largest_tensors(tensor_shapes),
        "txt_in": {
            "key": txt_in_key,
            "exists": txt_in_key is not None,
            "shape": txt_in_shape,
            "expected_prompt_embeds_shape": list(expected_prompt_embeds_shape),
            "expected_prompt_width": expected_hidden,
            "compatible_with_prompt_embeds": txt_in_compatible,
        },
        "safe_open_cpu": _safe_open_cpu(checkpoint),
    }


def _key_prefix(key: str) -> str:
    return key.split(".", 1)[0] if "." in key else key


def _find_key(tensors: dict[str, Any], target: str) -> str | None:
    if target in tensors:
        return target
    suffix = "." + target
    for key in sorted(tensors):
        if key.endswith(suffix):
            return key
    return None


def _interesting_keys(tensors: dict[str, Any]) -> list[str]:
    names = [
        "txt_in.weight",
        "txt_in.bias",
        "img_in.weight",
        "time_in.in_layer.weight",
        "guidance_in.in_layer.weight",
        "vector_in.in_layer.weight",
        "final_layer.linear.weight",
    ]
    found = []
    for name in names:
        key = _find_key(tensors, name)
        if key:
            found.append(key)
    return found


def _numel(shape: Any) -> int | None:
    if not isinstance(shape, list):
        return None
    total = 1
    for dim in shape:
        try:
            total *= int(dim)
        except (TypeError, ValueError):
            return None
    return total


def _tensor_size_bytes(info: dict[str, Any]) -> int | None:
    numel = _numel(info.get("shape", []))
    if numel is None:
        return None
    dtype = str(info.get("dtype", "unknown"))
    item_size = DTYPE_BYTES.get(dtype)
    if item_size is None:
        return None
    return numel * item_size


def _largest_tensors(tensor_shapes: dict[str, dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    rows = [
        {"key": key, **summary}
        for key, summary in tensor_shapes.items()
        if summary.get("size_bytes") is not None
    ]
    return sorted(rows, key=lambda item: int(item["size_bytes"]), reverse=True)[:limit]


def _safe_open_cpu(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        return {"ok": True, "key_count": len(keys), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "key_count": None, "error": repr(exc)}
