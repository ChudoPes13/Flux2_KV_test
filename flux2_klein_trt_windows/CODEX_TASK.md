# Task Contract

Build a Windows host + NVIDIA TensorRT-LLM Docker container test project for FLUX.2 Klein 9B-KV using ApacheOne NVFP4 checkpoints and TensorRT / TensorRT-LLM VisualGen as the runtime target.

Hard rules:

- Docker is allowed only for the explicit NVIDIA TensorRT-LLM runtime container because `tensorrt-llm` is not available as a native Windows package.
- No WSL.
- No Linux-only workflow outside the approved Docker runtime.
- No ComfyUI.
- No Telegram.
- No queues.
- No multi-worker orchestration.
- No batch generation.
- No frontend.
- No training.
- No ModelOpt conversion pipeline.
- No silent Diffusers fallback.
- No silent prompt-text generation when cached prompt embeddings are required.

The generation script must use cached prompt tensors and normalized image inputs. If TensorRT-LLM VisualGen cannot accept external prompt embeddings or cannot load the ApacheOne checkpoint layout in the Docker runtime, it must stop and write a diagnostic JSON.

Generation modes:

- `visualgen_prompt_text`: temporary smoke-test mode using public TensorRT-LLM VisualGen prompt text API. It must attempt `output.png`, set `prompt_cache_used=false`, and set `smoke_test_only=true`.
- `cached_embeddings_strict`: target mode using `prompt_embeds` and `text_ids` from `prompt_tensors.safetensors`. It must never fall back to prompt text.

Current implementation priority:

1. Complete end-to-end skeleton.
2. Strict environment and file checks.
3. Prompt cache writer using Diffusers Flux2 prompt-encoding helpers when local BFL tokenizer/text_encoder components are available.
4. TensorRT-LLM VisualGen adapter with runtime introspection and explicit unsupported-interface diagnostics.
5. Manual model download paths and optional controlled download flags.
