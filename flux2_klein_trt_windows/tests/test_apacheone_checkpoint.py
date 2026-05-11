from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file

from flux_trt.checkpoint_inspection import inspect_safetensors_checkpoint


def test_inspect_checkpoint_accepts_txt_in_width(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_file(
        {
            "txt_in.weight": torch.zeros((4, 16), dtype=torch.float16),
            "img_in.weight": torch.zeros((4, 8), dtype=torch.float16),
        },
        str(checkpoint),
    )

    report = inspect_safetensors_checkpoint(
        checkpoint,
        expected_prompt_embeds_shape=[1, 8, 16],
    )

    assert report["tensor_count"] == 2
    assert report["txt_in"]["exists"] is True
    assert report["txt_in"]["compatible_with_prompt_embeds"] is True
    assert report["txt_in"]["shape"] == [4, 16]


def test_inspect_checkpoint_reports_missing_txt_in(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_file({"other.weight": torch.zeros((2, 2), dtype=torch.float32)}, str(checkpoint))

    report = inspect_safetensors_checkpoint(
        checkpoint,
        expected_prompt_embeds_shape=[1, 8, 16],
    )

    assert report["txt_in"]["exists"] is False
    assert report["txt_in"]["compatible_with_prompt_embeds"] is False
