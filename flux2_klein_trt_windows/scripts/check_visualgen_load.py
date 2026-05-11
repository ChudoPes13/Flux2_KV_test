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
from flux_trt.runtime_layout import prepare_visualgen_runtime_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Check TensorRT-LLM VisualGen model loading only.")
    parser.add_argument("--variant", required=True, choices=["full", "txtattn_bf16"])
    args = parser.parse_args()

    config = load_project_config()
    config.ensure_directories()
    report_path = config.output_path("diagnostics") / f"check_visualgen_load_{args.variant}.json"

    report = {
        "stage": "check_visualgen_load",
        "status": "pending",
        "variant": args.variant,
        "created_at": utc_timestamp(),
        "target_gpu": "RTX 50 / Blackwell-class GPU for NVFP4 validation",
        "rtx_3070_target_note": (
            "RTX 3070 / Ampere is not the target GPU for NVFP4 acceptance. "
            "Use this machine only for CPU/IO/layout diagnostics."
        ),
        "runtime_layout": None,
        "visualgen_generate_signature": None,
        "visualgen_args_signature": None,
        "cuda_before_load": _cuda_before_load(),
        "worker_stdout_path": None,
        "worker_stderr_path": None,
        "worker_stdout_tail": None,
        "worker_stderr_tail": None,
        "detected_oom": False,
        "detected_unsupported_arch": False,
        "cuda_capability": None,
        "is_blackwell_or_newer": False,
        "nvfp4_target_gpu": False,
        "load_time_sec": None,
        "error": None,
        "traceback": None,
        "env": collect_env_info(allow_container=bool(config.runtime.get("allow_docker", False))),
    }
    report["cuda_capability"] = report["cuda_before_load"].get("cuda_capability")
    report["is_blackwell_or_newer"] = bool(
        report["cuda_before_load"].get("is_blackwell_or_newer", False)
    )
    report["nvfp4_target_gpu"] = bool(report["cuda_before_load"].get("nvfp4_target_gpu", False))

    visual_gen = None
    stdout_path = config.output_path("diagnostics") / f"check_visualgen_load_{args.variant}_stdout.log"
    stderr_path = config.output_path("diagnostics") / f"check_visualgen_load_{args.variant}_stderr.log"
    report["worker_stdout_path"] = str(stdout_path)
    report["worker_stderr_path"] = str(stderr_path)
    try:
        layout = prepare_visualgen_runtime_dir(config, args.variant)
        report["runtime_layout"] = layout
        if layout["missing"]:
            raise FileNotFoundError(
                "Cannot load VisualGen runtime_dir; missing files: "
                + "; ".join(layout["missing"])
            )

        with _capture_fds(stdout_path, stderr_path):
            import tensorrt  # noqa: F401
            from tensorrt_llm import VisualGen, VisualGenArgs

            report["visualgen_generate_signature"] = str(inspect.signature(VisualGen.generate))
            report["visualgen_args_signature"] = str(inspect.signature(VisualGenArgs))

            start = time.perf_counter()
            visual_gen = VisualGen(str(Path(layout["runtime_dir"])), args=VisualGenArgs(skip_warmup=True))
        report["load_time_sec"] = round(time.perf_counter() - start, 3)
        report["status"] = "success"
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
        print(f"visualgen load report: {report_path}")
        print(f"status: {report['status']}")
        if report["error"]:
            print(f"error: {report['error']}")

    return return_code


def _cuda_before_load() -> dict:
    try:
        import torch

        info = {
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_index": None,
            "gpu_name": None,
            "capability": None,
            "vram_total_bytes": None,
            "vram_free_bytes": None,
            "vram_total_gb": None,
            "vram_free_gb": None,
            "cuda_capability": None,
            "is_rtx_50_target_gpu": False,
            "is_rtx_3070": False,
            "is_blackwell_or_newer": False,
            "nvfp4_target_gpu": False,
        }
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            name = str(props.name)
            capability = list(torch.cuda.get_device_capability(device_index))
            info.update(
                {
                    "device_index": device_index,
                    "gpu_name": name,
                    "capability": capability,
                    "cuda_capability": capability,
                    "vram_total_bytes": int(total_bytes),
                    "vram_free_bytes": int(free_bytes),
                    "vram_total_gb": round(total_bytes / (1024**3), 3),
                    "vram_free_gb": round(free_bytes / (1024**3), 3),
                    "is_rtx_50_target_gpu": "RTX 50" in name.upper() or capability[0] >= 12,
                    "is_rtx_3070": "3070" in name.upper(),
                    "is_blackwell_or_newer": capability[0] >= 12,
                    "nvfp4_target_gpu": capability[0] >= 12,
                }
            )
        return info
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


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
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


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
    oom_patterns = [
        "out of memory",
        "cuda oom",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
    ]
    unsupported_arch_patterns = [
        "unsupported gpu architecture",
        "no kernel image is available",
        "invalid device function",
        "not supported on this device",
        "unsupported architecture",
    ]
    report["detected_oom"] = any(pattern in combined for pattern in oom_patterns)
    report["detected_unsupported_arch"] = any(
        pattern in combined for pattern in unsupported_arch_patterns
    )


if __name__ == "__main__":
    raise SystemExit(main())
