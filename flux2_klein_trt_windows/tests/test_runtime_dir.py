from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import ProjectConfig
from flux_trt.runtime_validation import validate_runtime_variant


def test_validate_runtime_variant_prepares_expected_layout(tmp_path: Path) -> None:
    config = _make_runtime_config(tmp_path)

    result = validate_runtime_variant(config, "full")

    assert result["status"] == "ok"
    assert result["gpu_loaded"] is False
    runtime_dir = Path(result["runtime_layout"]["runtime_dir"])
    assert (runtime_dir / "model_index.json").exists()
    assert (runtime_dir / "scheduler" / "scheduler_config.json").exists()
    assert (runtime_dir / "vae" / "diffusion_pytorch_model.safetensors").exists()
    assert (runtime_dir / "transformer" / "config.json").exists()
    assert (runtime_dir / "transformer" / "diffusion_pytorch_model.safetensors").exists()


def test_validate_runtime_variant_reports_missing_checkpoint(tmp_path: Path) -> None:
    config = _make_runtime_config(tmp_path)
    config.checkpoint_path("txtattn_bf16").unlink()

    result = validate_runtime_variant(config, "txtattn_bf16")

    assert result["status"] == "error"
    assert any("checkpoint" in error for error in result["errors"])


def _make_runtime_config(root: Path) -> ProjectConfig:
    _write(root / "models/bfl/model_index.json", "{}")
    _write(root / "models/bfl/scheduler/scheduler_config.json", "{}")
    _write(root / "models/bfl/tokenizer/tokenizer_config.json", "{}")
    _write(root / "models/bfl/tokenizer/tokenizer.json", "{}")
    _write(root / "models/bfl/vae/config.json", "{}")
    _write(root / "models/bfl/vae/diffusion_pytorch_model.safetensors", "vae")
    _write(root / "models/bfl/transformer/config.json", "{}")
    _write(root / "models/experimental/text_encoder/aifeifei_4bit/config.json", "{}")
    _write(root / "models/experimental/text_encoder/aifeifei_4bit/model.safetensors", "text")
    _write(root / "models/apacheone/full.safetensors", "full")
    _write(root / "models/apacheone/txtattn.safetensors", "txt")

    raw = {
        "models": {
            "bfl_dir": "models/bfl",
            "apacheone_dir": "models/apacheone",
            "experimental_dir": "models/experimental",
        },
        "checkpoints": {
            "full": "models/apacheone/full.safetensors",
            "txtattn_bf16": "models/apacheone/txtattn.safetensors",
        },
        "cache": {
            "root": "data/cache",
            "prompt_dir": "data/cache/prompt/main_prompt",
            "user_photo_dir": "data/cache/images/user_photo",
            "logo_dir": "data/cache/images/logo",
            "kv_dir": "data/cache/kv",
        },
        "output": {"root": "data/output", "diagnostics": "data/diagnostics"},
        "experimental": {
            "text_encoder": {
                "default_variant": "aifeifei_4bit",
                "variants": {
                    "aifeifei_4bit": {
                        "local_dir": "models/experimental/text_encoder/aifeifei_4bit",
                        "cache_dir": "data/cache/prompt/main_prompt_aifeifei_4bit",
                    }
                },
            }
        },
    }
    return ProjectConfig(root=root, raw=raw)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
