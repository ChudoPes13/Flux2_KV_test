# Current Status

Дата фиксации: 2026-05-11.

## Принятые проверки

- `scripts/validate_runtime_dir.py`: ok
- `scripts/inspect_apacheone_checkpoint.py`: `full` ok, `txtattn_bf16` ok
- `prompt_embeds`: `[1, 512, 12288]`, `torch.bfloat16`
- `text_ids`: `[1, 512, 4]`, `torch.int64`
- `scripts/mock_low_level_adapter_test.py`: ok
- `pytest`: 8 passed

## GPU статус

RTX 3070 не является целевой GPU для NVFP4 acceptance. На RTX 3070 проверяются только CPU/IO/layout/cache/diagnostics задачи.

Actual TensorRT-LLM VisualGen load/generation pending RTX 50 / Blackwell-class GPU.

Первый реальный runtime test делать на RTX 5060 Ti / RTX 5090 или другой Blackwell-class GPU с поддержкой нужного NVFP4 пути.

`scripts/rtx50_first_run_check.py` запускать только внутри NVIDIA TensorRT-LLM Docker container на RTX 50 / Blackwell GPU. Не использовать его как финальный тест на Windows host вне контейнера.

Порядок первого RTX 50 запуска:

1. `scripts/00_container_check.py`
2. `validate_runtime_dir.py`
3. `inspect_apacheone_checkpoint.py`
4. `mock_low_level_adapter_test.py`
5. `check_visualgen_supported_model.py`
6. `check_visualgen_load.py --variant full`
7. `check_visualgen_load.py --variant txtattn_bf16`
8. `visualgen_prompt_text` smoke-test
9. `cached_embeddings_strict` strict-test

## Архитектурный статус

`visualgen_prompt_text` остается smoke-test only режимом через public TensorRT-LLM VisualGen prompt-text path.

`cached_embeddings_strict` остается целевым режимом. Он обязан использовать cached tensors и не имеет права fallback на prompt text. Текущий public VisualGen API на этой установке не подтвердил поддержку external prompt embeddings; следующий blocker находится в low-level VisualGen/Flux2 adapter, ApacheOne single-file checkpoint loader или external embeddings support.
