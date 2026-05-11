from __future__ import annotations

import argparse
import gc
import inspect
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import load_project_config
from flux_trt.diagnostics import write_diagnostic
from flux_trt.prompt_cache import build_prompt_metadata, save_prompt_cache


def _select_dtype(torch_module: Any, value: str) -> Any:
    if value == "bf16":
        return torch_module.bfloat16
    if value == "fp16":
        return torch_module.float16
    is_bf16_supported = getattr(torch_module.cuda, "is_bf16_supported", lambda: False)
    if torch_module.cuda.is_available() and is_bf16_supported():
        return torch_module.bfloat16
    return torch_module.float16


def _variant_config(config: Any, text_encoder_variant: str) -> dict[str, Any]:
    variants = (
        config.raw.get("experimental", {})
        .get("text_encoder", {})
        .get("variants", {})
    )
    variant_config = variants.get(text_encoder_variant)
    if not variant_config:
        raise ValueError(f"Unknown text encoder variant: {text_encoder_variant}")
    return variant_config


def _resolve_prompt_cache_dir(
    config: Any,
    text_encoder_variant: str,
    override: str | None,
) -> Path:
    if override:
        return config.resolve_path(override)
    return config.prompt_cache_dir_for_variant(text_encoder_variant)


def _resolve_bfl_subdir(
    config: Any,
    subdir: str,
    required_file: str,
    allow_hub: bool,
) -> tuple[str, str | None, bool]:
    local_dir = config.model_dir("bfl_dir") / subdir
    if (local_dir / required_file).exists():
        return str(local_dir), None, True
    if allow_hub:
        return str(config.models["base_repo"]), subdir, False
    raise FileNotFoundError(
        f"Missing local BFL {subdir} file: {local_dir / required_file}. "
        "Download BFL companion files manually or rerun with --allow-hub."
    )


def _resolve_text_encoder_source(
    config: Any,
    text_encoder_variant: str,
    allow_hub: bool,
) -> tuple[str, str | None, bool]:
    if text_encoder_variant == "official":
        return _resolve_bfl_subdir(config, "text_encoder", "config.json", allow_hub)

    variant_config = _variant_config(config, text_encoder_variant)
    local_dir = config.resolve_path(str(variant_config["local_dir"]))
    if (local_dir / "config.json").exists():
        return str(local_dir), None, True
    if allow_hub:
        return str(variant_config["repo"]), None, False
    raise FileNotFoundError(
        f"Missing local experimental text encoder config: {local_dir / 'config.json'}. "
        "Download it manually or rerun with --allow-hub."
    )


def _with_subfolder(kwargs: dict[str, Any], subfolder: str | None) -> dict[str, Any]:
    if subfolder:
        kwargs = dict(kwargs)
        kwargs["subfolder"] = subfolder
    return kwargs


def _load_tokenizer(config: Any, allow_hub: bool) -> tuple[Any, str, bool]:
    source, subfolder, local_files_only = _resolve_bfl_subdir(
        config,
        "tokenizer",
        "tokenizer_config.json",
        allow_hub,
    )
    kwargs = _with_subfolder({"local_files_only": local_files_only}, subfolder)
    try:
        from transformers import AutoProcessor

        tokenizer = AutoProcessor.from_pretrained(source, **kwargs)
    except Exception:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    return tokenizer, f"{source}/{subfolder}" if subfolder else source, local_files_only


def _load_text_encoder(
    config: Any,
    text_encoder_variant: str,
    allow_hub: bool,
    dtype: Any,
    device: str,
) -> tuple[Any, str, bool]:
    source, subfolder, local_files_only = _resolve_text_encoder_source(
        config,
        text_encoder_variant,
        allow_hub,
    )
    common_kwargs = _with_subfolder({"local_files_only": local_files_only}, subfolder)

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(source, **common_kwargs)
    model_type = str(getattr(model_config, "model_type", "")).lower()
    if model_type == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        model_cls = Mistral3ForConditionalGeneration
    else:
        from transformers import AutoModelForCausalLM

        model_cls = AutoModelForCausalLM

    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        **common_kwargs,
    }
    if text_encoder_variant != "official" and device == "cuda":
        load_kwargs["device_map"] = "auto"

    text_encoder = model_cls.from_pretrained(source, **load_kwargs)
    text_encoder.eval()
    if "device_map" not in load_kwargs:
        text_encoder.to(device)
    return text_encoder, f"{source}/{subfolder}" if subfolder else source, local_files_only


def _flux2_pipeline_class() -> Any:
    import diffusers

    pipeline_cls = getattr(diffusers, "Flux2Pipeline", None)
    if pipeline_cls is None:
        pipeline_cls = getattr(diffusers, "Flux2KleinKVPipeline", None)
    if pipeline_cls is None:
        raise RuntimeError("Current diffusers package does not expose Flux2Pipeline.")
    if not hasattr(pipeline_cls, "_get_mistral_3_small_prompt_embeds"):
        raise RuntimeError(
            "Current diffusers Flux2 pipeline does not expose the prompt embedding helper."
        )
    if not hasattr(pipeline_cls, "_prepare_text_ids"):
        raise RuntimeError("Current diffusers Flux2 pipeline does not expose _prepare_text_ids.")
    return pipeline_cls


def _get_qwen_prompt_embeds(
    *,
    text_encoder: Any,
    tokenizer: Any,
    prompt: str,
    dtype: Any,
    device: Any,
    max_sequence_length: int = 512,
    hidden_states_layers: tuple[int, ...] = (10, 20, 30),
) -> Any:
    import torch
    from diffusers.pipelines.flux2.pipeline_flux2 import SYSTEM_MESSAGE

    messages_batch = [
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt.replace("[IMG]", "")},
        ]
    ]
    inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = output.hidden_states
    missing_layers = [layer for layer in hidden_states_layers if layer >= len(hidden_states)]
    if missing_layers:
        raise RuntimeError(
            f"Requested hidden state layers are unavailable: {missing_layers}. "
            f"Model returned {len(hidden_states)} hidden states."
        )
    out = torch.stack([hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=dtype, device=device)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    return out.permute(0, 2, 1, 3).reshape(
        batch_size,
        seq_len,
        num_channels * hidden_dim,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encode data/input/prompt.txt into safetensors cache."
    )
    parser.add_argument("--allow-hub", action="store_true", help="Allow Hugging Face downloads.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
    parser.add_argument(
        "--text-encoder-variant",
        default=None,
        choices=["official", "aifeifei_4bit"],
        help="Defaults to experimental.text_encoder.default_variant from config.",
    )
    parser.add_argument(
        "--output-cache-dir",
        default=None,
        help="Optional prompt cache output directory. Defaults to config variant cache_dir.",
    )
    args = parser.parse_args()

    config = load_project_config()
    config.ensure_directories()
    diagnostics_dir = config.output_path("diagnostics")
    text_encoder_variant = args.text_encoder_variant or config.default_text_encoder_variant()

    try:
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")

        prompt_path = config.input_path("prompt_path")
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise RuntimeError(f"Prompt file is empty: {prompt_path}")

        dtype = _select_dtype(torch, args.dtype)
        pipeline_cls = _flux2_pipeline_class()
        tokenizer, tokenizer_source, tokenizer_local_only = _load_tokenizer(
            config,
            args.allow_hub,
        )
        text_encoder, text_encoder_source, text_encoder_local_only = _load_text_encoder(
            config,
            text_encoder_variant,
            args.allow_hub,
            dtype,
            args.device,
        )

        with torch.inference_mode():
            if getattr(text_encoder.config, "model_type", None) == "qwen3":
                prompt_embeds = _get_qwen_prompt_embeds(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    dtype=dtype,
                    device=torch.device(args.device),
                    max_sequence_length=512,
                )
            else:
                prompt_embeds = pipeline_cls._get_mistral_3_small_prompt_embeds(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    dtype=dtype,
                    device=torch.device(args.device),
                    max_sequence_length=512,
                )
            text_ids = pipeline_cls._prepare_text_ids(prompt_embeds).to(args.device)

        tensors = {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
        }
        cache_dir = _resolve_prompt_cache_dir(
            config,
            text_encoder_variant,
            args.output_cache_dir,
        )
        encode_signature = (
            str(inspect.signature(pipeline_cls.encode_prompt))
            if hasattr(pipeline_cls, "encode_prompt")
            else None
        )
        metadata = build_prompt_metadata(
            prompt_id="main_prompt",
            prompt=prompt,
            dtype=str(dtype).replace("torch.", ""),
            text_encoder_source=text_encoder_source,
            tokenizer_source=tokenizer_source,
            extra={
                "text_encoder_variant": text_encoder_variant,
                "text_encoder_local_files_only": text_encoder_local_only,
                "tokenizer_local_files_only": tokenizer_local_only,
                "pipeline_class": f"{pipeline_cls.__module__}.{pipeline_cls.__name__}",
                "encoder_method": (
                    "local_qwen_prompt_embeds"
                    if getattr(text_encoder.config, "model_type", None) == "qwen3"
                    else "Flux2Pipeline._get_mistral_3_small_prompt_embeds"
                ),
                "encode_prompt_signature": encode_signature,
                "max_sequence_length": 512,
                "text_encoder_model_type": getattr(text_encoder.config, "model_type", None),
            },
        )
        saved = save_prompt_cache(
            cache_dir=cache_dir,
            prompt=prompt,
            tensors=tensors,
            metadata=metadata,
        )

        del prompt_embeds
        del text_ids
        del text_encoder
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"prompt cache: {saved['tensors']}")
        print(f"metadata: {saved['metadata']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        path = write_diagnostic(
            diagnostics_dir,
            "encode_prompt",
            "Prompt encoding failed.",
            exc=exc,
        )
        print(f"prompt encoding failed; diagnostic: {path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
