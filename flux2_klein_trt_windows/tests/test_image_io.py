from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("PIL")

from PIL import Image

from flux_trt.image_io import normalize_logo, normalize_user_photo


def test_normalize_user_photo_center_crops_to_rgb(tmp_path: Path) -> None:
    source = tmp_path / "photo.png"
    Image.new("RGB", (1200, 800), (20, 40, 60)).save(source)

    meta = normalize_user_photo(source, tmp_path / "cache", size=128)

    output = Path(meta["normalized_path"])
    assert (output.parent / "image_meta.json").exists()
    with Image.open(output) as image:
        assert image.size == (128, 128)
        assert image.mode == "RGB"


def test_normalize_logo_preserves_alpha(tmp_path: Path) -> None:
    source = tmp_path / "logo.png"
    Image.new("RGBA", (300, 100), (255, 0, 0, 128)).save(source)

    meta = normalize_logo(source, tmp_path / "cache", size=64)

    output = Path(meta["normalized_path"])
    assert (output.parent / "image_meta.json").exists()
    with Image.open(output) as image:
        assert image.size == (64, 64)
        assert image.mode == "RGBA"
