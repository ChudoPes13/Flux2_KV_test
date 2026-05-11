from __future__ import annotations

import argparse
import contextlib
import inspect
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import load_project_config
from flux_trt.diagnostics import utc_timestamp, write_json
from flux_trt.env import collect_env_info


DEFAULT_MODEL_PATH = "black-forest-labs/FLUX.2-dev"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a TensorRT-LLM VisualGen-supported model path separately from ApacheOne/Klein-KV."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--force-non-target",
        action="store_true",
        help="Try loading even when the GPU is not Blackwell or newer.",
    )
    args = parser.parse_args()

    config = load_project_config()
    config.ensure_directories()
    diagnostics_dir = config.output_path("diagnostics")
    safe_name = _safe_name(args.model_path)
    report_path = diagnostics_dir / f"check_visualgen_supported_model_{safe_name}.json"
    stdout_path = diagnostics_dir / f"check_visualgen_supported_model_{safe_name}_stdout.log"
    stderr_path = diagnostics_dir / f"check_visualgen_supported_model_{safe_name}_stderr.log"

    env = collect_env_info(allow_container=bool(config.runtime.get("allow_docker", False)))
    report = {
        "stage": "check_visualgen_supported_model",
        "created_at": utc_timestamp(),
        "status": "pending",
        "model_path": args.model_path,
        "purpose": (
            "Separate TensorRT-LLM VisualGen support/runtime issues from "
            "ApacheOne/Klein-KV checkpoint layout issues."
        ),
        "cuda_capability": env.get("cuda_capability"),
        "is_blackwell_or_newer": bool(env.get("is_blackwell_or_newer", False)),
        "nvfp4_target_gpu": bool(env.get("nvfp4_target_gpu", False)),
        "force_non_target": args.force_non_target,
        "visualgen_generate_signature": None,
        "visualgen_args_signature": None,
        "load_time_sec": None,
        "worker_stdout_path": str(stdout_path),
        "worker_stderr_path": str(stderr_path),
        "worker_stdout_tail": None,
        "worker_stderr_tail": None,
        "detected_oom": False,
        "detected_unsupported_arch": False,
        "error": None,
        "traceback": None,
        "env": env,
    }

    if not report["nvfp4_target_gpu"] and not args.force_non_target:
        report["status"] = "skipped_non_target_gpu"
        report["error"] = (
            "Supported VisualGen model load skipped because this GPU is not Blackwell or newer. "
            "Run on RTX 5060 Ti/5090 or pass --force-non-target."
        )
        write_json(report_path, report)
        print(f"supported model report: {report_path}")
        print(f"status: {report['status']}")
        return 1

    visual_gen = None
    try:
        with _capture_fds(stdout_path, stderr_path):
            import tensorrt  # noqa: F401
            from tensorrt_llm import VisualGen, VisualGenArgs

            report["visualgen_generate_signature"] = str(inspect.signature(VisualGen.generate))
            report["visualgen_args_signature"] = str(inspect.signature(VisualGenArgs))

            start = time.perf_counter()
            visual_gen = VisualGen(args.model_path, args=VisualGenArgs(skip_warmup=True))
            report["load_time_sec"] = round(time.perf_counter() - start, 3)
        report["status"] = "ok"
        return_code = 0
    except Exception as exc:  # noqa: BLE001
        report["status"] = "error"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        return_code = 1
    finally:
        if visual_gen is not None:
            try:
                visual_gen.shutdown()
            except Exception:  # noqa: BLE001
                pass
        report["worker_stdout_tail"] = _tail_text(stdout_path)
        report["worker_stderr_tail"] = _tail_text(stderr_path)
        _set_error_flags(report)
        write_json(report_path, report)
        print(f"supported model report: {report_path}")
        print(f"status: {report['status']}")
        if report["error"]:
            print(f"error: {report['error']}")

    return return_code


@contextlib.contextmanager
def _capture_fds(stdout_path: Path, stderr_path: Path) -> Iterator[None]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_handle, stderr_path.open(
            "w", encoding="utf-8", buffering=1
        ) as stderr_handle:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_handle.fileno(), 1)
            os.dup2(stderr_handle.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_stdout_fd, 1)
                os.dup2(saved_stderr_fd, 2)
    finally:
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def _tail_text(path: Path, limit: int = 12000) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def _set_error_flags(report: dict) -> None:
    combined = "\n".join(
        str(value or "")
        for value in [
            report.get("error"),
            report.get("traceback"),
            report.get("worker_stdout_tail"),
            report.get("worker_stderr_tail"),
        ]
    ).lower()
    report["detected_oom"] = any(
        pattern in combined
        for pattern in [
            "out of memory",
            "cuda oom",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
        ]
    )
    report["detected_unsupported_arch"] = any(
        pattern in combined
        for pattern in [
            "unsupported gpu architecture",
            "no kernel image is available",
            "invalid device function",
            "not supported on this device",
            "unsupported architecture",
        ]
    )


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


if __name__ == "__main__":
    raise SystemExit(main())
