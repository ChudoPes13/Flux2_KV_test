from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from flux_trt.prompt_cache import build_prompt_metadata, load_prompt_cache, save_prompt_cache


def test_prompt_cache_round_trip(tmp_path: Path) -> None:
    prompt = "A test prompt"
    tensors = {
        "prompt_embeds": torch.zeros((1, 4, 8), dtype=torch.float32),
        "text_ids": torch.arange(4, dtype=torch.int64),
    }
    metadata = build_prompt_metadata(
        prompt_id="main_prompt",
        prompt=prompt,
        dtype="fp32",
        text_encoder_source="local",
        tokenizer_source="local",
    )

    saved = save_prompt_cache(
        cache_dir=tmp_path,
        prompt=prompt,
        tensors=tensors,
        metadata=metadata,
    )
    loaded_tensors, loaded_meta = load_prompt_cache(tmp_path)

    assert saved["tensors"].exists()
    assert sorted(loaded_tensors) == ["prompt_embeds", "text_ids"]
    assert list(loaded_tensors["prompt_embeds"].shape) == [1, 4, 8]
    assert str(loaded_tensors["text_ids"].dtype) == "torch.int64"
    assert loaded_meta["prompt_id"] == "main_prompt"
