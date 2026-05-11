$ErrorActionPreference = "Stop"

$HostWorkspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$ContainerWorkspace = "/workspace/Flux2kv_test"
$Image = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14"
$HfCache = "C:\ai25\hf_cache"

New-Item -ItemType Directory -Force -Path $HfCache | Out-Null

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm `
  -p 8000:8000 `
  -e FLUX_ALLOW_DOCKER=1 `
  -e FLUX_DOCKER_IMAGE=$Image `
  -e HF_TOKEN=$env:HF_TOKEN `
  -e HF_HUB_ENABLE_HF_TRANSFER=1 `
  -v "${HostWorkspace}:${ContainerWorkspace}" `
  -v "${HfCache}:/root/.cache/huggingface" `
  $Image /bin/bash
