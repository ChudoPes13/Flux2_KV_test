# FLUX.2 Klein 9B-KV NVFP4 Windows Test Project

Локальный диагностируемый проект для Windows 11 host + NVIDIA TensorRT-LLM Docker container и ручной проверки FLUX.2 Klein 9B-KV NVFP4 через TensorRT / TensorRT-LLM VisualGen.

Проект намеренно не использует WSL, ComfyUI, Telegram, очереди, batch или Diffusers fallback. Docker разрешен только как явный NVIDIA TensorRT-LLM runtime, потому что `tensorrt-llm` не доступен как нормальный native Windows package. Если текущий TensorRT-LLM runtime не поддерживает нужный путь, скрипты сохраняют JSON diagnostic и останавливаются.

## Структура

```text
data/input/prompt.txt
data/input/user_photo.png
data/input/logo.png
models/apacheone/
models/bfl/
data/cache/
data/output/
data/diagnostics/
```

## Создание окружения

Основной runtime теперь контейнерный. Windows venv полезен только для легких проверок, подготовки input/cache и редактирования.

```powershell
cd C:\ai25\Flux2kv_test\flux2_klein_trt_windows

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-windows.txt
```

Если PyTorch CUDA wheels не находятся автоматически, установите PyTorch отдельно с актуального Windows CUDA index из документации PyTorch, затем повторите `pip install -r requirements-windows.txt`.

## Зависимости

Базовый список без жесткой фиксации версий:

```text
torch
torchvision
torchaudio
diffusers
transformers
accelerate
safetensors
huggingface_hub
hf_transfer
pillow
numpy
pyyaml
pydantic
psutil
rich
tqdm
bitsandbytes
triton-windows
pytest
```

`bitsandbytes` и `triton-windows` ставятся в Windows `.venv` для 4-bit text encoder cache. TensorRT-LLM ставится и используется внутри NVIDIA container. На native Windows `tensorrt-llm` не ожидается.

В container requirements `bitsandbytes` тоже включен, но его назначение узкое: он нужен только если `visualgen_prompt_text` пытается загрузить experimental `aifeifei_4bit` text encoder внутри public VisualGen prompt-text path. Это не улучшает `cached_embeddings_strict`, не решает ApacheOne single-file checkpoint loader и не добавляет поддержку external prompt embeddings.

## Docker TensorRT-LLM

Запуск контейнера с host workspace:

```powershell
cd C:\ai25\Flux2kv_test\flux2_klein_trt_windows
.\scripts\docker_enter.ps1
```

Эквивалент вручную:

```powershell
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm `
  -p 8000:8000 `
  -e FLUX_ALLOW_DOCKER=1 `
  -e HF_HUB_ENABLE_HF_TRANSFER=1 `
  -v C:\ai25\Flux2kv_test:/workspace/Flux2kv_test `
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14 /bin/bash
```

Внутри контейнера:

```bash
cd /workspace/Flux2kv_test/flux2_klein_trt_windows
python -m pip install -r requirements-container.txt
python scripts/00_env_check.py
python scripts/02_encode_prompt.py
python scripts/03_prepare_inputs.py
python scripts/validate_runtime_dir.py
python scripts/inspect_apacheone_checkpoint.py
python scripts/mock_low_level_adapter_test.py
```

`trtllm-serve` endpoint сам по себе не решает задачу cached prompt embeddings: public serving path обычно принимает prompt text. Для этого проекта важен offline Python VisualGen/API path. Если external prompt embeddings не поддерживаются, generation пишет diagnostic и останавливается.

Первый реальный runtime test запускать на RTX 5060 Ti / RTX 5090 или другой Blackwell-class GPU:

```bash
python scripts/rtx50_first_run_check.py
```

Этот скрипт выполняет env check, CPU/layout проверки, VisualGen load check для `full` и `txtattn_bf16`, затем `visualgen_prompt_text` smoke-test и `cached_embeddings_strict` strict-test. На не-Blackwell GPU runtime/generation шаги по умолчанию пропускаются; для диагностики можно явно передать `--force-non-target`, но это не является acceptance path.

## Ручная загрузка моделей

Большие файлы проект сам не скачивает по умолчанию. Команда ниже только печатает план и пишет `data/diagnostics/download_plan.json`:

```powershell
python scripts\01_download_models.py
```

ApacheOne NVFP4 checkpoints положить сюда:

```text
models/apacheone/flux2-klein-9b-kv-nvfp4.safetensors
models/apacheone/flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors
```

Ссылки:

- [ApacheOne repo](https://huggingface.co/ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed)
- [full checkpoint](https://huggingface.co/ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed/resolve/main/flux2-klein-9b-kv-nvfp4.safetensors)
- [txtattn BF16 checkpoint](https://huggingface.co/ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed/resolve/main/flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors)

BFL companion repo положить в:

```text
models/bfl/
```

Ожидаемый layout:

```text
models/bfl/
  model_index.json
  scheduler/
    scheduler_config.json
  tokenizer/
    added_tokens.json
    chat_template.jinja
    merges.txt
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
  transformer/
    config.json
  vae/
    config.json
    diffusion_pytorch_model.safetensors
```

BFL `text_encoder/`, BFL transformer weights и root file `flux-2-klein-9b-kv.safetensors` для текущего ApacheOne TensorRT path не нужны и намеренно не скачиваются. Нужен только `transformer/config.json`; основной transformer лежит в `models/apacheone/`.

Файлы брать из:

- [black-forest-labs/FLUX.2-klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv)

Репозиторий BFL gated: перед скачиванием нужно принять условия на Hugging Face. Если `huggingface-cli login` уже выполнен и доступ принят, можно использовать опциональную загрузку:

```powershell
python scripts\01_download_models.py --download --apacheone
python scripts\01_download_models.py --download --bfl-companion
```

Флаг `--bfl-companion` скачивает tokenizer, text_encoder, scheduler, VAE и `model_index.json`, но исключает BFL transformer и root single-file checkpoint.

## Экспериментальные 4-bit компоненты

4-bit text encoder можно положить отдельно, не ломая основной cache:

```text
models/experimental/text_encoder/aifeifei_4bit/
  config.json
  generation_config.json
  model.safetensors
```

Загрузка плана:

```powershell
python scripts\01_download_models.py
python scripts\01_download_models.py --download --experimental-text-encoder
```

Экспериментальный cache создается отдельно:

```powershell
python scripts\02_encode_prompt.py --text-encoder-variant aifeifei_4bit
```

Выход:

```text
data/cache/prompt/main_prompt_aifeifei_4bit/
```

`OzzyGT/flux2_klein_9B_bnb_4bit_transformer` - это 4-bit transformer, не text encoder. Его нельзя подставлять в `02_encode_prompt.py`; он оставлен только как отдельный экспериментальный transformer reference.

## Входные файлы

Положите изображения сюда:

```text
data/input/user_photo.png
data/input/logo.png
```

`user_photo.png` будет приведен к RGB 1024x1024 через center-crop + resize. `logo.png` будет приведен к 512x512 с сохранением alpha-channel при наличии прозрачности.

Промт уже создан:

```text
data/input/prompt.txt
```

## Запуск

```powershell
python scripts\00_env_check.py
python scripts\01_download_models.py
python scripts\02_encode_prompt.py
python scripts\03_prepare_inputs.py

python scripts\validate_runtime_dir.py
python scripts\inspect_apacheone_checkpoint.py
python scripts\mock_low_level_adapter_test.py

python scripts\check_visualgen_load.py --variant full
python scripts\check_visualgen_load.py --variant txtattn_bf16

python scripts\04_generate_once.py --variant full --mode cached_embeddings_strict
python scripts\04_generate_once.py --variant txtattn_bf16 --mode cached_embeddings_strict

python scripts\04_generate_once.py --variant full --mode visualgen_prompt_text

python scripts\05_compare_variants.py
```

`cached_embeddings_strict` не использует prompt text для generation step. Он требует готовый cache из `02_encode_prompt.py`. Если текущий VisualGen API не принимает external prompt embeddings, будет создан diagnostic с сообщением:

```text
External prompt embeddings are not supported by the current VisualGen interface on this setup.
```

`visualgen_prompt_text` - временный smoke-test режим. Он использует prompt text через public TensorRT-LLM VisualGen API, пишет `prompt_cache_used=false` и `smoke_test_only=true` в `run_report.json`, и не считается целевой архитектурой.

До запуска на RTX 50 / Blackwell-class GPU `output.png` не является acceptance criteria. На RTX 3070 этот проект используется только для CPU/IO/layout/cache/diagnostic проверок. Скрипт `check_visualgen_load.py` явно пишет `cuda_capability`, `is_blackwell_or_newer`, `nvfp4_target_gpu`, free VRAM before load, `detected_oom`, `detected_unsupported_arch` и предупреждение, что RTX 3070 не является целевой GPU для NVFP4.

Первый реальный runtime test делать на RTX 5060 Ti / RTX 5090:

```bash
python scripts/rtx50_first_run_check.py
```

Единый отчет:

```text
data/diagnostics/rtx50_first_run_report.json
```

## Режимы генерации

`visualgen_prompt_text`:

- smoke-test only;
- использует public TensorRT-LLM `VisualGen.generate()` / prompt-text path;
- может использовать prompt text;
- обязан попытаться сохранить `output.png` только на целевой машине, где VisualGen runtime загружается;
- пишет `mode=visualgen_prompt_text`, `prompt_cache_used=false`, `smoke_test_only=true`.

`cached_embeddings_strict`:

- целевой режим;
- использует `data/cache/prompt/main_prompt_aifeifei_4bit/prompt_tensors.safetensors`;
- требует `prompt_embeds [1,512,12288]` и `text_ids [1,512,4]`;
- не кодирует prompt text заново;
- не делает fallback на prompt text;
- если public VisualGen API не принимает external prompt embeddings, пишет diagnostic JSON и останавливается.

## CPU/IO диагностика

```powershell
python scripts\validate_runtime_dir.py
python scripts\validate_runtime_dir.py --variant full
python scripts\validate_runtime_dir.py --variant txtattn_bf16

python scripts\inspect_apacheone_checkpoint.py
python scripts\inspect_apacheone_checkpoint.py --variant full
python scripts\inspect_apacheone_checkpoint.py --variant txtattn_bf16

python scripts\mock_low_level_adapter_test.py
```

`validate_runtime_dir.py` проверяет `model_index.json`, scheduler, tokenizer, VAE, `transformer/config.json`, checkpoint path и готовит VisualGen runtime layout без загрузки модели в GPU.

`inspect_apacheone_checkpoint.py` читает safetensors header на CPU, считает tensors/dtypes/key prefixes, пишет key shapes и проверяет `txt_in.weight` на совместимость с шириной `prompt_embeds=12288`.

`mock_low_level_adapter_test.py` загружает cached prompt tensors, normalized user photo/logo, собирает `GenerationInputs`, проверяет shapes/dtypes и не запускает реальную генерацию.

Для отделения проблем TensorRT-LLM VisualGen от проблем ApacheOne/Klein-KV есть отдельная проверка официального VisualGen-compatible model path:

```bash
python scripts/check_visualgen_supported_model.py
```

По умолчанию используется `black-forest-labs/FLUX.2-dev`. Скрипт не использует ApacheOne runtime layout и пишет отдельный diagnostic report в `data/diagnostics/`.

## Проверки разработки

```powershell
python -m compileall src scripts tests
pytest
```

## Диагностика

Все ошибки пишутся в:

```text
data/diagnostics/diagnostic_YYYYMMDD_HHMMSS.json
```

Environment report:

```text
data/diagnostics/env_report.json
```

Run outputs:

```text
data/output/run_YYYYMMDD_HHMMSS/
```
