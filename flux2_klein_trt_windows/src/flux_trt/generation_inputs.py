from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .prompt_cache import load_prompt_cache
from .tensor_io import tensor_summary


EXPECTED_PROMPT_EMBEDS_SHAPE = [1, 512, 12288]
EXPECTED_TEXT_IDS_SHAPE = [1, 512, 4]
EXPECTED_USER_PHOTO_SIZE = [1024, 1024]
EXPECTED_LOGO_SIZE = [512, 512]


@dataclass(frozen=True)
class GenerationInputs:
    prompt_cache_dir: Path
    prompt_meta: dict[str, Any]
    prompt_tensors: dict[str, Any]
    user_photo_path: Path
    user_photo: dict[str, Any]
    logo_path: Path
    logo: dict[str, Any]
    width: int
    height: int
    steps: int
    seed: int
    guidance_scale: float

    def summary(self) -> dict[str, Any]:
        return {
            "prompt_cache_dir": str(self.prompt_cache_dir),
            "prompt_meta": self.prompt_meta,
            "prompt_tensors": tensor_summary(self.prompt_tensors),
            "user_photo_path": str(self.user_photo_path),
            "user_photo": self.user_photo,
            "logo_path": str(self.logo_path),
            "logo": self.logo,
            "generation": {
                "width": self.width,
                "height": self.height,
                "steps": self.steps,
                "seed": self.seed,
                "guidance_scale": self.guidance_scale,
            },
        }


def load_generation_inputs(
    config: ProjectConfig,
    *,
    text_encoder_variant: str | None = None,
) -> GenerationInputs:
    prompt_cache_dir = config.prompt_cache_dir_for_variant(text_encoder_variant)
    prompt_tensors, prompt_meta = load_prompt_cache(prompt_cache_dir)

    user_photo_path = config.cache_path("user_photo_dir") / "normalized_1024.png"
    logo_path = config.cache_path("logo_dir") / "normalized_512.png"

    return GenerationInputs(
        prompt_cache_dir=prompt_cache_dir,
        prompt_meta=prompt_meta,
        prompt_tensors=prompt_tensors,
        user_photo_path=user_photo_path,
        user_photo=_image_summary(user_photo_path),
        logo_path=logo_path,
        logo=_image_summary(logo_path),
        width=int(config.generation["width"]),
        height=int(config.generation["height"]),
        steps=int(config.generation["steps"]),
        seed=int(config.generation["seed"]),
        guidance_scale=float(config.generation.get("guidance_scale", 4.0)),
    )


def validate_generation_inputs(inputs: GenerationInputs) -> dict[str, Any]:
    errors = []
    tensor_info = tensor_summary(inputs.prompt_tensors)

    prompt_embeds = tensor_info.get("prompt_embeds")
    text_ids = tensor_info.get("text_ids")
    if prompt_embeds is None:
        errors.append("prompt_tensors.safetensors missing prompt_embeds")
    elif prompt_embeds["shape"] != EXPECTED_PROMPT_EMBEDS_SHAPE:
        errors.append(
            f"prompt_embeds shape {prompt_embeds['shape']} != {EXPECTED_PROMPT_EMBEDS_SHAPE}"
        )
    if prompt_embeds is not None and prompt_embeds["dtype"] != "torch.bfloat16":
        errors.append(f"prompt_embeds dtype {prompt_embeds['dtype']} != torch.bfloat16")

    if text_ids is None:
        errors.append("prompt_tensors.safetensors missing text_ids")
    elif text_ids["shape"] != EXPECTED_TEXT_IDS_SHAPE:
        errors.append(f"text_ids shape {text_ids['shape']} != {EXPECTED_TEXT_IDS_SHAPE}")
    if text_ids is not None and text_ids["dtype"] != "torch.int64":
        errors.append(f"text_ids dtype {text_ids['dtype']} != torch.int64")

    if inputs.user_photo["size"] != EXPECTED_USER_PHOTO_SIZE:
        errors.append(f"user_photo size {inputs.user_photo['size']} != {EXPECTED_USER_PHOTO_SIZE}")
    if inputs.logo["size"] != EXPECTED_LOGO_SIZE:
        errors.append(f"logo size {inputs.logo['size']} != {EXPECTED_LOGO_SIZE}")

    return {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "generation_inputs": inputs.summary(),
        "real_generation_started": False,
    }


def _image_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing normalized image: {path}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pillow.") from exc

    with Image.open(path) as image:
        return {
            "path": str(path),
            "size": [int(image.width), int(image.height)],
            "mode": image.mode,
        }
