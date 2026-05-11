from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import load_project_config
from flux_trt.diagnostics import utc_timestamp, write_json
from flux_trt.env import collect_env_info, running_in_container


def main() -> int:
    parser = argparse.ArgumentParser(description="Check NVIDIA TensorRT-LLM Docker container readiness.")
    parser.add_argument(
        "--allow-non-target-gpu",
        action="store_true",
        help="Do not fail when the GPU is not Blackwell or newer.",
    )
    args = parser.parse_args()

    config = load_project_config()
    config.ensure_directories()
    env = collect_env_info(allow_container=True)

    errors = list(env.get("errors", []))
    warnings = list(env.get("warnings", []))
    if not running_in_container():
        errors.append("Not running inside the NVIDIA TensorRT-LLM Docker container.")
    if not env.get("nvfp4_target_gpu") and not args.allow_non_target_gpu:
        errors.append("Current GPU is not Blackwell-or-newer; it is not an NVFP4 target GPU.")
    if env.get("status") != "ok":
        errors.append("Base environment check failed.")

    report = {
        "stage": "container_check",
        "created_at": utc_timestamp(),
        "status": "ok" if not errors else "error",
        "errors": errors,
        "warnings": warnings,
        "gpu_name": env.get("gpu_name"),
        "cuda_capability": env.get("cuda_capability"),
        "is_blackwell_or_newer": env.get("is_blackwell_or_newer"),
        "nvfp4_target_gpu": env.get("nvfp4_target_gpu"),
        "vram_total_gb": env.get("vram_total_gb"),
        "vram_free_before_load": env.get("vram_free_before_load"),
        "docker_image": env.get("docker_image"),
        "torch_version": env.get("torch_cuda", {}).get("torch_version"),
        "tensorrt_llm_version": _import_version(env, "tensorrt_llm"),
        "model_dir": None,
        "variant": None,
        "mode": "container_check",
        "prompt_cache_used": None,
        "smoke_test_only": None,
        "detected_oom": False,
        "detected_unsupported_arch": False,
        "detected_missing_model_index": False,
        "detected_invalid_safetensors": False,
        "running_in_container": env.get("running_in_container"),
        "env": env,
    }

    report_path = config.output_path("diagnostics") / "container_report.json"
    write_json(report_path, report)
    print(f"container report: {report_path}")
    print(f"status: {report['status']}")
    for error in errors:
        print(f"  - {error}")
    return 0 if report["status"] == "ok" else 1


def _import_version(env: dict, module_name: str) -> str | None:
    for item in env.get("imports", []):
        if item.get("module") == module_name:
            return item.get("version")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
