from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import load_project_config
from flux_trt.diagnostics import write_diagnostic, write_json
from flux_trt.model_loader import (
    build_download_plan,
    download_apacheone,
    download_bfl_companion,
    download_experimental_text_encoder,
    download_experimental_transformer,
    validate_model_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print manual model download paths or optionally download selected files."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download selected model files. Default is dry-run.",
    )
    parser.add_argument(
        "--apacheone",
        action="store_true",
        help="With --download, download only the two ApacheOne checkpoint files.",
    )
    parser.add_argument(
        "--bfl-companion",
        action="store_true",
        help="With --download, download BFL companion metadata/config files without transformer weights.",
    )
    parser.add_argument(
        "--experimental-text-encoder",
        action="store_true",
        help="With --download, download optional aifeifei798 4-bit text encoder files.",
    )
    parser.add_argument(
        "--experimental-transformer",
        action="store_true",
        help="With --download, download optional OzzyGT 4-bit transformer files.",
    )
    args = parser.parse_args()

    config = load_project_config()
    config.ensure_directories()
    plan = build_download_plan(config)
    plan_path = write_json(config.output_path("diagnostics") / "download_plan.json", plan)

    print(f"download plan: {plan_path}")
    for item in plan["apacheone_files"]:
        print(f"{item['variant']}:")
        print(f"  url: {item['url']}")
        print(f"  target: {item['target_path']}")
    print(f"BFL companion repo: {plan['bfl_url']}")
    print(f"BFL target dir: {plan['bfl_target_dir']}")
    for item in plan["experimental"]:
        print(f"experimental {item['role']} ({item['variant']}):")
        print(f"  url: {item['url']}")
        print(f"  target: {item['target_dir']}")
        print(f"  note: {item['note']}")

    if not args.download:
        status = validate_model_files(config)
        write_json(config.output_path("diagnostics") / "model_file_status.json", status)
        print("dry-run only; no files downloaded")
        return 0

    if (
        not args.apacheone
        and not args.bfl_companion
        and not args.experimental_text_encoder
        and not args.experimental_transformer
    ):
        print("Refusing broad download. Use --download --apacheone and/or --download --bfl-companion.")
        return 2

    try:
        downloaded = []
        if args.apacheone:
            downloaded.extend(str(path) for path in download_apacheone(config))
        if args.bfl_companion:
            downloaded.append(str(download_bfl_companion(config)))
        if args.experimental_text_encoder:
            downloaded.append(str(download_experimental_text_encoder(config)))
        if args.experimental_transformer:
            downloaded.append(str(download_experimental_transformer(config)))
        status = validate_model_files(config)
        write_json(
            config.output_path("diagnostics") / "download_result.json",
            {"downloaded": downloaded, "status": status},
        )
        print("download complete")
        return 0
    except Exception as exc:  # noqa: BLE001
        path = write_diagnostic(
            config.output_path("diagnostics"),
            "download",
            "Model download failed.",
            exc=exc,
        )
        print(f"download failed; diagnostic: {path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
