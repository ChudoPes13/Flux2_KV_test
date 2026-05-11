from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ProjectConfig


APACHEONE_FILES = {
    "full": "flux2-klein-9b-kv-nvfp4.safetensors",
    "txtattn_bf16": "flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors",
}

APACHEONE_LINKS = {
    name: f"https://huggingface.co/ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed/resolve/main/{filename}"
    for name, filename in APACHEONE_FILES.items()
}

BFL_REPO_LINK = "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv"

EXPERIMENTAL_TEXT_ENCODER = {
    "variant": "aifeifei_4bit",
    "repo": "aifeifei798/FLUX.2-klein-9B-text_encoder-4bit",
    "url": "https://huggingface.co/aifeifei798/FLUX.2-klein-9B-text_encoder-4bit",
    "role": "text_encoder",
    "note": "Experimental 4-bit Qwen3 text encoder for prompt-cache tests.",
}

EXPERIMENTAL_TRANSFORMER = {
    "variant": "ozzygt_bnb_4bit",
    "repo": "OzzyGT/flux2_klein_9B_bnb_4bit_transformer",
    "url": "https://huggingface.co/OzzyGT/flux2_klein_9B_bnb_4bit_transformer",
    "role": "transformer",
    "note": "Experimental 4-bit transformer, not a text encoder.",
}

BFL_COMPANION_ALLOW_PATTERNS = [
    "model_index.json",
    "README.md",
    "LICENSE*",
    ".gitattributes",
    "scheduler/*",
    "tokenizer/*",
    "transformer/config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

BFL_COMPANION_IGNORE_PATTERNS = [
    "transformer/*.safetensors",
    "transformer/*.bin",
    "transformer/*.pt",
    "transformer/*.pth",
    "transformer_2/*",
    "flux-2-klein-9b-kv.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
]

TEXT_ENCODER_ALLOW_PATTERNS = [
    "README.md",
    "LICENSE.md",
    ".gitattributes",
    "config.json",
    "generation_config.json",
    "model.safetensors",
]

OZZY_TRANSFORMER_ALLOW_PATTERNS = [
    "README.md",
    "LICENSE.md",
    ".gitattributes",
    "config.json",
    "diffusion_pytorch_model.safetensors",
]


def build_download_plan(config: ProjectConfig) -> dict[str, Any]:
    apacheone_dir = config.model_dir("apacheone_dir")
    bfl_dir = config.model_dir("bfl_dir")
    experimental_dir = config.model_dir("experimental_dir")
    return {
        "manual_download_required": True,
        "apacheone_repo": config.models["apacheone_repo"],
        "apacheone_files": [
            {
                "variant": variant,
                "filename": filename,
                "url": APACHEONE_LINKS[variant],
                "target_path": str((apacheone_dir / filename).resolve()),
            }
            for variant, filename in APACHEONE_FILES.items()
        ],
        "bfl_repo": config.models["base_repo"],
        "bfl_url": BFL_REPO_LINK,
        "bfl_target_dir": str(bfl_dir.resolve()),
        "experimental": [
            {
                **EXPERIMENTAL_TEXT_ENCODER,
                "target_dir": str((experimental_dir / "text_encoder" / "aifeifei_4bit").resolve()),
                "primary_file": "model.safetensors",
            },
            {
                **EXPERIMENTAL_TRANSFORMER,
                "target_dir": str((experimental_dir / "transformer" / "ozzygt_bnb_4bit").resolve()),
                "primary_file": "diffusion_pytorch_model.safetensors",
            },
        ],
        "notes": [
            "Accept the gated BFL license on Hugging Face before downloading.",
            "The script defaults to dry-run to avoid accidental large downloads.",
            "Use --download --apacheone for only the two ApacheOne checkpoint files.",
            "Use --download --bfl-companion for tokenizer, scheduler, VAE, and model_index files.",
            "The BFL original text_encoder, transformer folder, and root 18.2 GB single-file checkpoint are intentionally excluded.",
            "Use --download --experimental-text-encoder for the active aifeifei 4-bit text encoder if it is missing.",
            "OzzyGT is an experimental 4-bit transformer, not a text encoder.",
        ],
    }


def validate_model_files(config: ProjectConfig, variant: str | None = None) -> dict[str, Any]:
    variants = [variant] if variant else sorted(APACHEONE_FILES)
    checkpoints = []
    for item in variants:
        path = config.checkpoint_path(item)
        checkpoints.append(
            {
                "variant": item,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    bfl_dir = config.model_dir("bfl_dir")
    model_index = bfl_dir / "model_index.json"
    experimental_dir = config.model_dir("experimental_dir")
    bfl_expected_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
        "transformer/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    return {
        "checkpoints": checkpoints,
        "bfl_dir": str(bfl_dir),
        "bfl_model_index_exists": model_index.exists(),
        "bfl_model_index_path": str(model_index),
        "bfl_expected_files": [
            {
                "path": str(bfl_dir / item),
                "exists": (bfl_dir / item).exists(),
            }
            for item in bfl_expected_files
        ],
        "bfl_original_text_encoder_present": (bfl_dir / "text_encoder").exists(),
        "experimental_text_encoder": {
            "variant": EXPERIMENTAL_TEXT_ENCODER["variant"],
            "path": str(experimental_dir / "text_encoder" / "aifeifei_4bit"),
            "model_exists": (experimental_dir / "text_encoder" / "aifeifei_4bit" / "model.safetensors").exists(),
        },
        "experimental_transformer": {
            "variant": EXPERIMENTAL_TRANSFORMER["variant"],
            "path": str(experimental_dir / "transformer" / "ozzygt_bnb_4bit"),
            "model_exists": (
                experimental_dir
                / "transformer"
                / "ozzygt_bnb_4bit"
                / "diffusion_pytorch_model.safetensors"
            ).exists(),
        },
    }


def download_apacheone(config: ProjectConfig) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Missing dependency: huggingface_hub.") from exc

    output_dir = config.model_dir("apacheone_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in APACHEONE_FILES.values():
        downloaded = hf_hub_download(
            repo_id=str(config.models["apacheone_repo"]),
            filename=filename,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        paths.append(Path(downloaded))
    return paths


def download_bfl_companion(config: ProjectConfig) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Missing dependency: huggingface_hub.") from exc

    output_dir = config.model_dir("bfl_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(config.models["base_repo"]),
        local_dir=str(output_dir),
        allow_patterns=BFL_COMPANION_ALLOW_PATTERNS,
        ignore_patterns=BFL_COMPANION_IGNORE_PATTERNS,
        local_dir_use_symlinks=False,
    )
    return output_dir


def download_experimental_text_encoder(config: ProjectConfig) -> Path:
    return _snapshot_download_to(
        repo_id=EXPERIMENTAL_TEXT_ENCODER["repo"],
        output_dir=config.model_dir("experimental_dir") / "text_encoder" / "aifeifei_4bit",
        allow_patterns=TEXT_ENCODER_ALLOW_PATTERNS,
    )


def download_experimental_transformer(config: ProjectConfig) -> Path:
    return _snapshot_download_to(
        repo_id=EXPERIMENTAL_TRANSFORMER["repo"],
        output_dir=config.model_dir("experimental_dir") / "transformer" / "ozzygt_bnb_4bit",
        allow_patterns=OZZY_TRANSFORMER_ALLOW_PATTERNS,
    )


def _snapshot_download_to(
    *,
    repo_id: str,
    output_dir: Path,
    allow_patterns: list[str],
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Missing dependency: huggingface_hub.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )
    return output_dir
