from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import load_project_config
from flux_trt.diagnostics import write_diagnostic
from flux_trt.env import collect_env_info
from flux_trt.pipeline_adapter import FluxTrtPipelineAdapter
from flux_trt.report import base_run_report, copy_if_exists, create_run_dir, write_run_report


MODES = ["cached_embeddings_strict", "visualgen_prompt_text"]


def run_once(
    variant: str,
    output_dir: Path | None = None,
    mode: str = "cached_embeddings_strict",
) -> tuple[int, Path]:
    config = load_project_config()
    config.ensure_directories()
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")

    if output_dir is None:
        run_dir = create_run_dir(config.output_path("root"), "run")
    else:
        run_dir = output_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.generation["seed"])
    width = int(config.generation["width"])
    height = int(config.generation["height"])
    steps = int(config.generation["steps"])
    checkpoint = config.checkpoint_path(variant)
    report = base_run_report(
        variant=variant,
        apacheone_checkpoint=str(checkpoint),
        base_repo=str(config.models["base_repo"]),
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        mode=mode,
    )
    if mode == "visualgen_prompt_text":
        report["prompt_cache_used"] = False
        report["user_photo_cache_used"] = False
        report["logo_cache_used"] = False
        report["smoke_test_only"] = True
    else:
        report["prompt_cache_used"] = True
        report["smoke_test_only"] = False

    total_start = time.perf_counter()
    try:
        env = collect_env_info(allow_container=bool(config.runtime.get("allow_docker", False)))
        torch_cuda = env.get("torch_cuda", {})
        report["gpu_name"] = torch_cuda.get("gpu_name")
        report["vram_total_gb"] = torch_cuda.get("vram_total_gb")
        if env["status"] != "ok":
            raise RuntimeError("Environment is not ready: " + "; ".join(env["errors"]))

        adapter = FluxTrtPipelineAdapter(config)
        load_start = time.perf_counter()
        adapter.load(variant)
        report["load_time_sec"] = round(time.perf_counter() - load_start, 3)

        generation_start = time.perf_counter()
        if mode == "visualgen_prompt_text":
            prompt_text = config.input_path("prompt_path").read_text(encoding="utf-8").strip()
            output_png = adapter.generate_from_prompt_text(
                prompt_text=prompt_text,
                output_dir=str(run_dir),
                seed=seed,
                width=width,
                height=height,
                steps=steps,
            )
        else:
            output_png = adapter.generate_from_cached_inputs(
                prompt_cache_dir=str(config.prompt_cache_dir_for_variant()),
                user_photo_path=str(config.cache_path("user_photo_dir") / "normalized_1024.png"),
                logo_path=str(config.cache_path("logo_dir") / "normalized_512.png"),
                output_dir=str(run_dir),
                seed=seed,
                width=width,
                height=height,
                steps=steps,
            )
        report["generation_time_sec"] = round(time.perf_counter() - generation_start, 3)
        report["output_path"] = output_png
        report["status"] = "success"
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        report["status"] = "error"
        report["error"] = str(exc)
        write_diagnostic(
            config.output_path("diagnostics"),
            "generate",
            "Generation failed.",
            exc=exc,
            extra={
                "variant": variant,
                "mode": mode,
                "run_dir": str(run_dir),
                "apacheone_checkpoint": str(checkpoint),
            },
        )
        exit_code = 1
    finally:
        report["total_time_sec"] = round(time.perf_counter() - total_start, 3)
        _add_vram_peak(report)
        if mode == "visualgen_prompt_text":
            copy_if_exists(config.input_path("prompt_path"), run_dir / "used_prompt.txt")
        else:
            copy_if_exists(
                config.prompt_cache_dir_for_variant() / "prompt.txt",
                run_dir / "used_prompt.txt",
            )
            copy_if_exists(
                config.cache_path("user_photo_dir") / "normalized_1024.png",
                run_dir / "used_user_photo.png",
            )
            copy_if_exists(
                config.cache_path("logo_dir") / "normalized_512.png",
                run_dir / "used_logo.png",
            )
        write_run_report(run_dir, report)

    return exit_code, run_dir


def _add_vram_peak(report: dict) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            report["vram_peak_allocated_gb"] = round(
                torch.cuda.max_memory_allocated() / (1024**3), 3
            )
    except Exception:  # noqa: BLE001
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one TensorRT-LLM VisualGen generation.")
    parser.add_argument("--variant", required=True, choices=["full", "txtattn_bf16"])
    parser.add_argument(
        "--mode",
        default="cached_embeddings_strict",
        choices=MODES,
        help="Strict cached embeddings target mode or prompt-text smoke test mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory, used by compare_variants.",
    )
    args = parser.parse_args()

    code, run_dir = run_once(
        args.variant,
        Path(args.output_dir) if args.output_dir else None,
        args.mode,
    )
    print(f"run dir: {run_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
