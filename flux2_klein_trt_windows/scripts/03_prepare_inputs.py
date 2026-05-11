from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import load_project_config
from flux_trt.diagnostics import write_diagnostic
from flux_trt.image_io import normalize_logo, normalize_user_photo


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize user photo and logo inputs.")
    parser.parse_args()

    config = load_project_config()
    config.ensure_directories()
    try:
        user_meta = normalize_user_photo(
            config.input_path("user_photo_path"),
            config.cache_path("user_photo_dir"),
            int(config.generation["width"]),
        )
        logo_meta = normalize_logo(
            config.input_path("logo_path"),
            config.cache_path("logo_dir"),
            int(config.generation["logo_width"]),
        )
        print(f"user photo: {user_meta['normalized_path']}")
        print(f"logo: {logo_meta['normalized_path']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        path = write_diagnostic(
            config.output_path("diagnostics"),
            "prepare_inputs",
            "Input image preparation failed.",
            exc=exc,
        )
        print(f"input preparation failed; diagnostic: {path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

