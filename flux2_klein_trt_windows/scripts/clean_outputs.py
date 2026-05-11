from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import load_project_config


def _safe_clear(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if root_resolved not in [resolved, *resolved.parents]:
        raise RuntimeError(f"Refusing to delete outside project root: {resolved}")
    if not resolved.exists():
        return
    for item in resolved.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean generated outputs.")
    parser.add_argument("--yes", action="store_true", help="Actually delete files.")
    parser.add_argument("--diagnostics", action="store_true", help="Also clear diagnostic JSON files.")
    args = parser.parse_args()

    config = load_project_config()
    output_root = config.output_path("root")
    diagnostics_root = config.output_path("diagnostics")

    print(f"output root: {output_root}")
    if args.diagnostics:
        print(f"diagnostics root: {diagnostics_root}")
    if not args.yes:
        print("dry-run only; pass --yes to delete generated files")
        return 0

    _safe_clear(output_root, config.root)
    if args.diagnostics:
        _safe_clear(diagnostics_root, config.root)
    print("clean complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

