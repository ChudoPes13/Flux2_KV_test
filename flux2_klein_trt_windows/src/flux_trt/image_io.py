from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .diagnostics import utc_timestamp, write_json
from .hashing import sha256_file


def _resample_filter() -> Any:
    from PIL import Image

    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _has_alpha(image: Any) -> bool:
    if image.mode in ("RGBA", "LA"):
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return "A" in image.getbands()


def normalize_user_photo(source_path: str | Path, output_dir: str | Path, size: int = 1024) -> dict[str, Any]:
    return _normalize_image(
        source_path=source_path,
        output_dir=output_dir,
        image_id="user_photo",
        output_name="normalized_1024.png",
        width=size,
        height=size,
        mode_policy="rgb_center_crop",
    )


def normalize_logo(source_path: str | Path, output_dir: str | Path, size: int = 512) -> dict[str, Any]:
    return _normalize_image(
        source_path=source_path,
        output_dir=output_dir,
        image_id="logo",
        output_name="normalized_512.png",
        width=size,
        height=size,
        mode_policy="logo_contain",
    )


def _normalize_image(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    image_id: str,
    output_name: str,
    width: int,
    height: int,
    mode_policy: str,
) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pillow.") from exc

    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Missing input image: {source}")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_copy = target_dir / "source.png"
    shutil.copy2(source, source_copy)

    source_hash = sha256_file(source)
    with Image.open(source) as image:
        source_width, source_height = image.size
        if mode_policy == "rgb_center_crop":
            normalized = _center_crop_square(image).convert("RGB")
            normalized = normalized.resize((width, height), _resample_filter())
        elif mode_policy == "logo_contain":
            has_alpha = _has_alpha(image)
            target_mode = "RGBA" if has_alpha else "RGB"
            normalized = _contain_resize(image.convert(target_mode), width, height, target_mode)
        else:
            raise ValueError(f"Unknown image normalization policy: {mode_policy}")

        normalized_path = target_dir / output_name
        normalized.save(normalized_path, format="PNG")
        mode = normalized.mode

    metadata = {
        "image_id": image_id,
        "source_path": str(source.resolve()),
        "source_copy_path": str(source_copy.resolve()),
        "normalized_path": str(normalized_path.resolve()),
        "source_sha256": source_hash,
        "source_width": source_width,
        "source_height": source_height,
        "width": width,
        "height": height,
        "mode": mode,
        "created_at": utc_timestamp(),
    }
    write_json(target_dir / "image_meta.json", metadata)
    return metadata


def _center_crop_square(image: Any) -> Any:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def _contain_resize(image: Any, width: int, height: int, mode: str) -> Any:
    from PIL import Image

    working = image.copy()
    working.thumbnail((width, height), _resample_filter())
    if mode == "RGBA":
        background = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    else:
        background = Image.new("RGB", (width, height), (255, 255, 255))
    x = (width - working.width) // 2
    y = (height - working.height) // 2
    if mode == "RGBA":
        background.alpha_composite(working, (x, y))
    else:
        background.paste(working, (x, y))
    return background

