from __future__ import annotations

import runpy
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "flux2_klein_trt_windows"
SCRIPT = PROJECT_ROOT / "scripts" / "rtx50_first_run_check.py"


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    runpy.run_path(str(SCRIPT), run_name="__main__")
