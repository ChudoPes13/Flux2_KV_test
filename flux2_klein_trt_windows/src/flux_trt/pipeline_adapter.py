from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .diagnostics import write_diagnostic
from .prompt_cache import load_prompt_cache
from .runtime_layout import prepare_visualgen_runtime_dir, runtime_dir_for_variant


EXTERNAL_EMBEDDINGS_UNSUPPORTED = (
    "External prompt embeddings are not supported by the current VisualGen interface on this setup."
)


class UnsupportedVisualGenInterfaceError(RuntimeError):
    pass


class FluxTrtPipelineAdapter:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.variant: str | None = None
        self.checkpoint_path: Path | None = None
        self.visual_gen_cls: Any | None = None
        self.visual_gen_args_cls: Any | None = None
        self.visual_gen_params_cls: Any | None = None
        self.media_storage_cls: Any | None = None
        self.generate_signature: str | None = None
        self.load_time_sec: float | None = None

    def load(self, variant: str) -> None:
        start = time.perf_counter()
        self.variant = variant
        self.checkpoint_path = self.config.checkpoint_path(variant)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing ApacheOne checkpoint: {self.checkpoint_path}")

        bfl_model_index = self.config.model_dir("bfl_dir") / "model_index.json"
        if not bfl_model_index.exists():
            raise FileNotFoundError(
                f"Missing BFL companion model_index.json: {bfl_model_index}"
            )

        try:
            import tensorrt  # noqa: F401
            from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams
            from tensorrt_llm.serve.media_storage import MediaStorage
        except Exception as exc:  # noqa: BLE001
            write_diagnostic(
                self.config.output_path("diagnostics"),
                "generate",
                "TensorRT or TensorRT-LLM VisualGen import failed.",
                exc=exc,
                extra={
                    "variant": variant,
                    "apacheone_checkpoint": str(self.checkpoint_path),
                },
            )
            raise

        self.visual_gen_cls = VisualGen
        self.visual_gen_args_cls = VisualGenArgs
        self.visual_gen_params_cls = VisualGenParams
        self.media_storage_cls = MediaStorage
        self.generate_signature = str(inspect.signature(VisualGen.generate))
        self.load_time_sec = round(time.perf_counter() - start, 3)

    def generate_from_cached_inputs(
        self,
        prompt_cache_dir: str,
        user_photo_path: str,
        logo_path: str,
        output_dir: str,
        seed: int = 42,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
    ) -> str:
        if self.visual_gen_cls is None or self.visual_gen_params_cls is None:
            raise RuntimeError("Adapter is not loaded. Call load(variant) first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        prompt_tensors, prompt_meta = load_prompt_cache(prompt_cache_dir)
        user_photo = Path(user_photo_path)
        logo = Path(logo_path)
        if not user_photo.exists():
            raise FileNotFoundError(f"Missing normalized user photo: {user_photo}")
        if not logo.exists():
            raise FileNotFoundError(f"Missing normalized logo: {logo}")

        if not self._supports_external_prompt_embeddings():
            self._write_unsupported_diagnostic(
                prompt_cache_dir=prompt_cache_dir,
                user_photo_path=str(user_photo),
                logo_path=str(logo),
                prompt_meta=prompt_meta,
                prompt_tensor_names=sorted(prompt_tensors),
            )
            raise UnsupportedVisualGenInterfaceError(EXTERNAL_EMBEDDINGS_UNSUPPORTED)

        raise UnsupportedVisualGenInterfaceError(
            "A future VisualGen API appears to expose prompt embedding parameters, "
            "but this project has no verified implementation for image references "
            "and ApacheOne single-file checkpoint replacement yet."
        )

    def check_visualgen_load(self, variant: str) -> dict[str, Any]:
        if self.visual_gen_cls is None or self.visual_gen_args_cls is None:
            self.load(variant)

        layout = prepare_visualgen_runtime_dir(self.config, variant)
        if layout["missing"]:
            raise FileNotFoundError(
                "Cannot load VisualGen runtime_dir; missing files: "
                + "; ".join(layout["missing"])
            )

        runtime_dir = runtime_dir_for_variant(self.config, variant)
        visual_gen = None
        start = time.perf_counter()
        try:
            args = self.visual_gen_args_cls(skip_warmup=True)
            visual_gen = self.visual_gen_cls(str(runtime_dir), args=args)
            return {
                "runtime_dir": str(runtime_dir),
                "load_time_sec": round(time.perf_counter() - start, 3),
                "visualgen_generate_signature": self.generate_signature,
                "layout": layout,
            }
        finally:
            if visual_gen is not None:
                try:
                    visual_gen.shutdown()
                except Exception:
                    pass

    def generate_from_prompt_text(
        self,
        *,
        prompt_text: str,
        output_dir: str,
        seed: int = 42,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
    ) -> str:
        if self.variant is None:
            raise RuntimeError("Adapter is not loaded. Call load(variant) first.")
        if self.visual_gen_cls is None or self.visual_gen_args_cls is None or self.visual_gen_params_cls is None:
            raise RuntimeError("Adapter is not loaded. Call load(variant) first.")
        if not prompt_text.strip():
            raise ValueError("Prompt text is empty.")

        layout = prepare_visualgen_runtime_dir(self.config, self.variant)
        if layout["missing"]:
            raise FileNotFoundError(
                "Cannot run VisualGen prompt-text smoke test; missing files: "
                + "; ".join(layout["missing"])
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        runtime_dir = runtime_dir_for_variant(self.config, self.variant)

        visual_gen = None
        try:
            args = self.visual_gen_args_cls(skip_warmup=True)
            params = self.visual_gen_params_cls(
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=float(self.config.generation.get("guidance_scale", 4.0)),
                max_sequence_length=512,
                seed=seed,
                num_images_per_prompt=1,
            )
            visual_gen = self.visual_gen_cls(str(runtime_dir), args=args)
            media_output = visual_gen.generate(prompt_text, params=params)
            png_path = output_path / "output.png"
            self._save_media_output_image(media_output, png_path)
            return str(png_path)
        finally:
            if visual_gen is not None:
                try:
                    visual_gen.shutdown()
                except Exception:
                    pass

    @staticmethod
    def _save_media_output_image(media_output: Any, png_path: Path) -> None:
        image = getattr(media_output, "image", None)
        if image is None:
            raise RuntimeError("VisualGen returned no image tensor.")
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Saving VisualGen output requires torch and pillow.") from exc

        tensor = image
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim == 4:
                tensor = tensor[0]
            tensor = tensor.detach().cpu()
            if tensor.dtype != torch.uint8:
                tensor = tensor.clamp(0, 255).to(torch.uint8)
            array = tensor.numpy()
        else:
            array = image[0] if getattr(image, "ndim", 0) == 4 else image
        png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(png_path)

    def _supports_external_prompt_embeddings(self) -> bool:
        if self.visual_gen_cls is None:
            return False
        signature = inspect.signature(self.visual_gen_cls.generate)
        parameter_names = set(signature.parameters)
        explicit_names = {
            "prompt_embeds",
            "prompt_embeddings",
            "encoder_hidden_states",
            "text_embeddings",
            "cached_prompt",
            "cached_prompt_embeddings",
        }
        return bool(explicit_names.intersection(parameter_names))

    def _write_unsupported_diagnostic(
        self,
        *,
        prompt_cache_dir: str,
        user_photo_path: str,
        logo_path: str,
        prompt_meta: dict[str, Any],
        prompt_tensor_names: list[str],
    ) -> None:
        write_diagnostic(
            self.config.output_path("diagnostics"),
            "generate",
            EXTERNAL_EMBEDDINGS_UNSUPPORTED,
            extra={
                "variant": self.variant,
                "apacheone_checkpoint": str(self.checkpoint_path),
                "base_repo": self.config.models.get("base_repo"),
                "bfl_dir": str(self.config.model_dir("bfl_dir")),
                "prompt_cache_dir": prompt_cache_dir,
                "prompt_meta": prompt_meta,
                "prompt_tensor_names": prompt_tensor_names,
                "user_photo_path": user_photo_path,
                "logo_path": logo_path,
                "visualgen_generate_signature": self.generate_signature,
                "reason": (
                    "Official TensorRT-LLM VisualGen examples expose generation "
                    "with inputs=prompt text and params=VisualGenParams. This setup "
                    "did not expose a verified public parameter for external prompt embeddings."
                ),
                "persistent_reference_kv_cache": (
                    "Persistent reference KV-cache is not exposed by the current pipeline/runtime."
                ),
            },
        )
