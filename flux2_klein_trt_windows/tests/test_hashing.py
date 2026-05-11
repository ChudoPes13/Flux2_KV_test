from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.hashing import sha256_file, sha256_text


def test_sha256_text_matches_file(tmp_path: Path) -> None:
    text = "hello flux"
    path = tmp_path / "sample.txt"
    path.write_text(text, encoding="utf-8")

    assert sha256_text(text) == sha256_file(path)

