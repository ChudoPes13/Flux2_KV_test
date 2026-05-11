# Codex Project Spec: FLUX.2 Klein 9B-KV NVFP4 Test Project on Windows 11

## 0. Цель проекта

Создать локальный тестовый Python-проект для запуска **FLUX.2 Klein 9B-KV** на **Windows 11** с видеокартой **RTX 5060 Ti 16 GB** и **64 GB RAM**.

Проект нужен не для Telegram, не для веб-продукта и не для нагрузки. Сейчас цель — один максимально понятный тестовый пайплайн:

```text
1 фиксированный промт
+ 1 фото пользователя
+ 1 PNG-логотип
→ отдельный TextEncoder-сервис сохраняет prompt embeddings
→ generation script использует готовые embeddings + изображения
→ TensorRT / TensorRT-LLM VisualGen runtime
→ 1 итоговая картинка PNG
```

Работаем только с **готовыми ApacheOne NVFP4-весами**:

```text
ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed
```

Нужно попробовать обе версии:

```text
flux2-klein-9b-kv-nvfp4.safetensors
flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors
```

## 1. Жёсткие ограничения

Codex должен строго соблюдать эти правила:

```text
НЕ использовать Docker.
НЕ использовать WSL.
НЕ использовать Linux-инструкции.
НЕ использовать ComfyUI.
НЕ использовать Telegram.
НЕ делать очереди.
НЕ делать multi-worker.
НЕ делать batch-режим.
НЕ добавлять отдельный frontend.
НЕ добавлять обучение.
НЕ добавлять подготовку новых весов.
НЕ добавлять ModelOpt pipeline.
НЕ скачивать и не использовать лишние модели без необходимости.
```

Проект должен быть рассчитан на:

```text
OS: Windows 11 x64
GPU: NVIDIA RTX 5060 Ti 16 GB
RAM: 64 GB
Disk: local NVMe
Python: 3.12 preferred
Python 3.13: только если все нужные пакеты реально устанавливаются и проходят env-check
```

Для текущей задачи **выбрать Python 3.12 по умолчанию**. Python 3.13 оставить как экспериментальный вариант, но не делать его основным.

## 2. Важная техническая реальность

ApacheOne repo содержит готовые transformer-веса в `.safetensors`, но это не полноценная Diffusers-папка со всеми компонентами. Поэтому проекту всё равно нужны companion-компоненты из официального репозитория:

```text
black-forest-labs/FLUX.2-klein-9b-kv
```

Использовать официальный repo для:

```text
tokenizer
text_encoder
VAE
scheduler/config
pipeline metadata
model_index/config files
```

Но основной transformer для generation должен загружаться из ApacheOne:

```text
models/apacheone/flux2-klein-9b-kv-nvfp4.safetensors
или
models/apacheone/flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors
```

Если TensorRT-LLM VisualGen на Windows не сможет напрямую принять ApacheOne `.safetensors`, Codex должен:

```text
1. Не подменять задачу на ComfyUI.
2. Не подменять задачу на Docker.
3. Не делать молчаливый fallback.
4. Сохранить подробный diagnostic report:
   - какие пакеты установлены
   - какая версия CUDA/TensorRT/TensorRT-LLM
   - какой файл модели пробовали загрузить
   - какая точная ошибка возникла
   - на каком этапе pipeline остановился
```

## 3. Архитектура проекта

Нужны три независимых слоя:

```text
A. TextEncoder Cache Tool
B. Input Preparation Tool
C. Generation Tool
```

### A. TextEncoder Cache Tool

Назначение:

```text
Взять один фиксированный промт из файла.
Прогнать его через text_encoder.
Сохранить все нужные prompt tensors на NVMe.
После этого generation step не должен заново кодировать текст.
```

Промт:

```text
1 фиксированный промт
5–7 предложений
обычный текстовый файл UTF-8
```

TextEncoder tool должен сохранить не просто один tensor наугад, а **все tensors, которые реально нужны pipeline**.

Codex должен проверить актуальную сигнатуру `Flux2KleinKVPipeline.encode_prompt()` или эквивалентного метода и сохранить возвращаемые данные в структурированном виде.

Ожидаемый формат сохранения:

```text
data/cache/prompt/main_prompt/
  prompt.txt
  prompt_meta.json
  prompt_tensors.safetensors
```

`prompt_meta.json` должен содержать:

```json
{
  "prompt_id": "main_prompt",
  "prompt_sha256": "...",
  "text_encoder_source": "black-forest-labs/FLUX.2-klein-9b-kv",
  "tokenizer_source": "black-forest-labs/FLUX.2-klein-9b-kv",
  "dtype": "bf16_or_fp16",
  "created_at": "...",
  "python_version": "...",
  "torch_version": "...",
  "transformers_version": "...",
  "diffusers_version": "..."
}
```

### B. Input Preparation Tool

Назначение:

```text
Принять 1 фото пользователя и 1 PNG-логотип.
Привести их к фиксированным размерам.
Сохранить нормализованные копии и metadata.
```

Входы:

```text
data/input/user_photo.png
data/input/logo.png
```

Размеры:

```text
user_photo: 1024x1024
logo: 512x512
output image: 1024x1024
```

Если входное фото не квадратное, использовать аккуратный center-crop + resize.

Если логотип PNG с прозрачностью, сохранить alpha-channel.

Ожидаемый формат:

```text
data/cache/images/user_photo/
  source.png
  normalized_1024.png
  image_meta.json

data/cache/images/logo/
  source.png
  normalized_512.png
  image_meta.json
```

`image_meta.json` должен содержать:

```json
{
  "image_id": "user_photo",
  "source_path": "...",
  "normalized_path": "...",
  "source_sha256": "...",
  "width": 1024,
  "height": 1024,
  "mode": "RGB_or_RGBA",
  "created_at": "..."
}
```

### C. Generation Tool

Назначение:

```text
Загрузить выбранный ApacheOne transformer checkpoint.
Загрузить companion pipeline config/tokenizer/VAE/scheduler из BFL repo.
Загрузить prompt embeddings из cache.
Загрузить normalized user photo + logo.
Запустить одну генерацию.
Сохранить PNG и подробный run report.
```

CLI должен поддерживать два режима выбора ApacheOne-весов:

```powershell
python scripts\generate_once.py --variant full
python scripts\generate_once.py --variant txtattn_bf16
```

Где:

```text
full:
  flux2-klein-9b-kv-nvfp4.safetensors

txtattn_bf16:
  flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors
```

Выходы:

```text
data/output/run_YYYYMMDD_HHMMSS/
  output.png
  run_report.json
  used_prompt.txt
  used_user_photo.png
  used_logo.png
```

`run_report.json` должен содержать:

```json
{
  "variant": "full_or_txtattn_bf16",
  "apacheone_checkpoint": "...",
  "base_repo": "black-forest-labs/FLUX.2-klein-9b-kv",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "steps": 4,
  "device": "cuda",
  "gpu_name": "...",
  "vram_total_gb": 16,
  "vram_peak_allocated_gb": null,
  "load_time_sec": null,
  "generation_time_sec": null,
  "total_time_sec": null,
  "prompt_cache_used": true,
  "user_photo_cache_used": true,
  "logo_cache_used": true,
  "status": "success_or_error",
  "error": null
}
```

## 4. TensorRT / TensorRT-LLM VisualGen требования

Проект должен использовать TensorRT/TensorRT-LLM VisualGen как основной runtime target.

Но Codex должен учесть важное ограничение:

```text
Стандартный trtllm-serve image endpoint обычно принимает prompt text,
а не заранее сохранённые prompt embeddings.
```

Поэтому в проекте нужны два слоя:

```text
1. Low-level pipeline adapter.
2. Public test script.
```

### Low-level pipeline adapter

Файл:

```text
src/flux_trt/pipeline_adapter.py
```

Задачи:

```text
- инкапсулировать загрузку TensorRT/TensorRT-LLM VisualGen
- инкапсулировать загрузку ApacheOne checkpoint
- предоставить метод generate_from_cached_inputs(...)
- не заставлять main generation script заново кодировать prompt text
```

Интерфейс:

```python
class FluxTrtPipelineAdapter:
    def __init__(self, config: ProjectConfig):
        ...

    def load(self, variant: str) -> None:
        ...

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
        ...
```

Если текущая версия TensorRT-LLM VisualGen не поддерживает подачу внешних embeddings напрямую, adapter должен явно вернуть ошибку:

```text
External prompt embeddings are not supported by the current VisualGen interface on this setup.
```

И должен создать diagnostic report.

Запрещено:

```text
Нельзя молча использовать prompt text вместо prompt embeddings.
Нельзя молча запускать Diffusers вместо TensorRT.
Нельзя молча игнорировать logo input.
```

## 5. KV-cache и хранение

На текущем этапе всё храним постоянно на локальном NVMe.

Хранить:

```text
prompt text
prompt tensors
source user photo
normalized user photo
source logo
normalized logo
run outputs
run reports
diagnostics
```

Папка:

```text
data/cache/
```

Стратегия удаления пока не нужна.

### Важное правило по KV-cache

Если pipeline/runtime предоставляет реальный доступ к reference KV-cache, сохранить его:

```text
data/cache/kv/user_photo/
data/cache/kv/logo/
```

Если runtime не предоставляет публичный доступ к reference KV-cache, Codex должен:

```text
1. Не придумывать фейковый KV-cache.
2. Сохранить только доступные реальные артефакты:
   - prompt tensors
   - normalized images
   - image hashes
   - output reports
3. В diagnostic report указать:
   "Persistent reference KV-cache is not exposed by the current pipeline/runtime."
```

## 6. Структура проекта

Создать такую структуру:

```text
flux2_klein_trt_windows/
  README.md
  CODEX_TASK.md
  requirements.txt
  requirements-windows.txt
  .env.example
  .gitignore

  configs/
    project.yaml

  data/
    input/
      prompt.txt
      user_photo.png
      logo.png
    cache/
      prompt/
      images/
      kv/
    output/
    diagnostics/

  models/
    apacheone/
      .gitkeep
    bfl/
      .gitkeep

  scripts/
    00_env_check.py
    01_download_models.py
    02_encode_prompt.py
    03_prepare_inputs.py
    04_generate_once.py
    05_compare_variants.py
    clean_outputs.py

  src/
    flux_trt/
      __init__.py
      config.py
      env.py
      hashing.py
      image_io.py
      prompt_cache.py
      tensor_io.py
      model_loader.py
      pipeline_adapter.py
      diagnostics.py
      report.py

  tests/
    test_hashing.py
    test_image_io.py
    test_prompt_cache.py
```

## 7. Конфиг проекта

Файл:

```text
configs/project.yaml
```

Пример:

```yaml
project:
  name: flux2_klein_trt_windows
  os: windows
  python: "3.12"
  device: cuda

models:
  base_repo: black-forest-labs/FLUX.2-klein-9b-kv
  apacheone_repo: ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed
  apacheone_dir: models/apacheone
  bfl_dir: models/bfl

checkpoints:
  full: models/apacheone/flux2-klein-9b-kv-nvfp4.safetensors
  txtattn_bf16: models/apacheone/flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors

input:
  prompt_path: data/input/prompt.txt
  user_photo_path: data/input/user_photo.png
  logo_path: data/input/logo.png

cache:
  root: data/cache
  prompt_dir: data/cache/prompt/main_prompt
  user_photo_dir: data/cache/images/user_photo
  logo_dir: data/cache/images/logo
  kv_dir: data/cache/kv

generation:
  width: 1024
  height: 1024
  logo_width: 512
  logo_height: 512
  steps: 4
  seed: 42
  batch_size: 1

output:
  root: data/output
  diagnostics: data/diagnostics

runtime:
  prefer_tensorrt: true
  prefer_visualgen: true
  allow_diffusers_fallback: false
  allow_prompt_text_in_generation: false
  allow_docker: false
  allow_wsl: false
```

## 8. Скрипты

### scripts/00_env_check.py

Проверить:

```text
Windows 11
Python version
CUDA availability
torch.cuda.is_available()
GPU name
VRAM total
NVIDIA driver
TensorRT import
TensorRT-LLM import
diffusers import
transformers import
safetensors import
```

Создать:

```text
data/diagnostics/env_report.json
```

Если TensorRT или TensorRT-LLM не установлены, вывести понятную ошибку и инструкцию, но не продолжать generation step.

### scripts/01_download_models.py

Назначение:

```text
Скачать нужные файлы моделей с Hugging Face.
```

Скачать:

```text
ApacheOne:
- flux2-klein-9b-kv-nvfp4.safetensors
- flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors

BFL:
- tokenizer/text_encoder/VAE/scheduler/config/model_index files, необходимые для pipeline
```

Важно:

```text
Не скачивать лишние большие transformer-веса BFL, если можно избежать.
Если библиотека требует полный BFL repo для инициализации, сначала попробовать snapshot_download с allow_patterns.
Если без полного repo pipeline не собирается, записать это в diagnostic report.
```

### scripts/02_encode_prompt.py

Назначение:

```text
Прочитать data/input/prompt.txt.
Получить prompt tensors через TextEncoder.
Сохранить их в data/cache/prompt/main_prompt.
```

Правила:

```text
- использовать UTF-8
- сохранять tensors в safetensors, где возможно
- metadata отдельно в JSON
- после завершения освобождать GPU memory
```

### scripts/03_prepare_inputs.py

Назначение:

```text
Прочитать data/input/user_photo.png и data/input/logo.png.
Сохранить normalized копии.
Сохранить image metadata.
```

Правила:

```text
- user_photo → RGB 1024x1024
- logo → RGBA 512x512, если есть alpha
- не терять прозрачность логотипа
- считать SHA256 исходных файлов
```

### scripts/04_generate_once.py

Назначение:

```text
Одна генерация.
```

CLI:

```powershell
python scripts\04_generate_once.py --variant full
python scripts\04_generate_once.py --variant txtattn_bf16
```

Pipeline:

```text
1. env check
2. load config
3. load selected ApacheOne checkpoint
4. load companion BFL configs/components
5. load prompt tensors from cache
6. load normalized user_photo
7. load normalized logo
8. run TensorRT/TensorRT-LLM VisualGen generation
9. save output.png
10. save run_report.json
```

### scripts/05_compare_variants.py

Назначение:

```text
Запустить обе версии ApacheOne весов на одном и том же входе.
Сохранить две картинки.
Сохранить compare_report.json.
```

CLI:

```powershell
python scripts\05_compare_variants.py
```

Выход:

```text
data/output/compare_YYYYMMDD_HHMMSS/
  full/output.png
  txtattn_bf16/output.png
  compare_report.json
```

## 9. requirements.txt

Codex должен создать `requirements.txt` и `requirements-windows.txt`.

Базово нужны:

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
```

Отдельно проверить установку:

```text
tensorrt
tensorrt-llm
```

Важно:

```text
Не фиксировать версии вслепую.
Сначала сделать env_check.
Если TensorRT-LLM на native Windows не устанавливается, diagnostic report должен явно это показать.
```

## 10. PowerShell команды

Создать `README.md` с командами:

```powershell
cd C:\ai25\flux2_klein_trt_windows

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

pip install -r requirements-windows.txt

python scripts\00_env_check.py
python scripts\01_download_models.py
python scripts\02_encode_prompt.py
python scripts\03_prepare_inputs.py

python scripts\04_generate_once.py --variant full
python scripts\04_generate_once.py --variant txtattn_bf16

python scripts\05_compare_variants.py
```

## 11. Пример input prompt

Создать пример файла:

```text
data/input/prompt.txt
```

Пример промта:

```text
Create a realistic professional advertising image based on the provided user photo and logo reference. Keep the person's identity, face structure, and natural proportions consistent with the reference photo. Dress the person in a clean modern construction work jacket with a premium commercial look. Place the provided logo naturally on the jacket or nearby branded element without distorting its shape. Use realistic studio lighting, sharp details, clean background, and a high-end promotional photography style. The final image must look like a polished product/service advertisement, not a cartoon or illustration.
```

## 12. Поведение при ошибках

Любая ошибка должна сохраняться в:

```text
data/diagnostics/
```

Формат:

```text
diagnostic_YYYYMMDD_HHMMSS.json
```

Обязательно писать:

```json
{
  "stage": "env_check_or_download_or_encode_prompt_or_prepare_inputs_or_generate",
  "status": "error",
  "message": "...",
  "traceback": "...",
  "python_version": "...",
  "torch_version": "...",
  "cuda_available": true,
  "gpu_name": "...",
  "vram_total_gb": 16
}
```

## 13. Запрещённое поведение Codex

Codex не должен:

```text
- заменять TensorRT на ComfyUI
- добавлять Docker
- добавлять WSL
- добавлять Telegram
- добавлять очередь задач
- добавлять несколько воркеров
- генерировать batch
- использовать prompt text в generation step, если уже есть prompt cache
- игнорировать logo.png
- игнорировать user_photo.png
- игнорировать выбранный variant
- скачивать произвольные модели без явной причины
- скрывать ошибки совместимости Windows/TensorRT
```

Если что-то не поддерживается текущими библиотеками, нужно сделать честный diagnostic report и остановиться.

## 14. Acceptance Criteria

Проект считается готовым, если:

```text
1. На Windows 11 создаётся venv Python 3.12.
2. scripts/00_env_check.py создаёт env_report.json.
3. scripts/01_download_models.py скачивает обе ApacheOne .safetensors.
4. scripts/02_encode_prompt.py создаёт prompt_tensors.safetensors и prompt_meta.json.
5. scripts/03_prepare_inputs.py создаёт normalized_1024.png для фото и normalized_512.png для логотипа.
6. scripts/04_generate_once.py принимает --variant full и --variant txtattn_bf16.
7. Каждый запуск создаёт отдельную папку в data/output.
8. При успехе сохраняется output.png.
9. При ошибке сохраняется diagnostic JSON с точной причиной.
10. Нигде не используется Docker, WSL, Telegram, ComfyUI, очередь или batch.
```

## 15. Первый приоритет реализации

Сначала реализовать минимальный end-to-end skeleton:

```text
env_check
config loader
image preprocessing
prompt cache writer
model file downloader/checker
generation adapter stub with real TensorRT/TensorRT-LLM import checks
diagnostic reporting
```

Потом подключить реальный TensorRT/TensorRT-LLM VisualGen generation path.

## 16. Второй приоритет реализации

После skeleton:

```text
1. Подключить загрузку BFL companion pipeline components.
2. Подключить замену transformer weights на ApacheOne variant.
3. Проверить, поддерживает ли текущий runtime external prompt tensors.
4. Запустить generate_once.
5. Запустить compare_variants.
```

## 17. Главное правило

Проект должен быть честным и диагностируемым.

Если на Windows 11 + RTX 5060 Ti 16 GB + native TensorRT/TensorRT-LLM какой-то этап не поддерживается текущими библиотеками, проект не должен притворяться, что всё работает. Он должен сохранить точную ошибку и показать, какой следующий технический шаг нужен.
