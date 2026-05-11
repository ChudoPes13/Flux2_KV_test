from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CONFIG_RELATIVE_PATH = Path("configs") / "project.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    raw: Mapping[str, Any]

    @property
    def project(self) -> Mapping[str, Any]:
        return self.raw.get("project", {})

    @property
    def models(self) -> Mapping[str, Any]:
        return self.raw.get("models", {})

    @property
    def checkpoints(self) -> Mapping[str, Any]:
        return self.raw.get("checkpoints", {})

    @property
    def input(self) -> Mapping[str, Any]:
        return self.raw.get("input", {})

    @property
    def cache(self) -> Mapping[str, Any]:
        return self.raw.get("cache", {})

    @property
    def generation(self) -> Mapping[str, Any]:
        return self.raw.get("generation", {})

    @property
    def output(self) -> Mapping[str, Any]:
        return self.raw.get("output", {})

    @property
    def runtime(self) -> Mapping[str, Any]:
        return self.raw.get("runtime", {})

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.root / path).resolve()

    def checkpoint_path(self, variant: str) -> Path:
        if variant not in self.checkpoints:
            known = ", ".join(sorted(self.checkpoints))
            raise ValueError(f"Unknown variant '{variant}'. Known variants: {known}")
        return self.resolve_path(str(self.checkpoints[variant]))

    def model_dir(self, key: str) -> Path:
        return self.resolve_path(str(self.models[key]))

    def input_path(self, key: str) -> Path:
        return self.resolve_path(str(self.input[key]))

    def cache_path(self, key: str) -> Path:
        return self.resolve_path(str(self.cache[key]))

    def default_text_encoder_variant(self) -> str:
        return str(
            self.raw.get("experimental", {})
            .get("text_encoder", {})
            .get("default_variant", "official")
        )

    def prompt_cache_dir_for_variant(self, variant: str | None = None) -> Path:
        selected = variant or self.default_text_encoder_variant()
        if selected == "official":
            return self.cache_path("prompt_dir")
        variants = (
            self.raw.get("experimental", {})
            .get("text_encoder", {})
            .get("variants", {})
        )
        variant_config = variants.get(selected)
        if not variant_config or "cache_dir" not in variant_config:
            raise ValueError(f"No prompt cache dir configured for text encoder variant: {selected}")
        return self.resolve_path(str(variant_config["cache_dir"]))

    def output_path(self, key: str) -> Path:
        return self.resolve_path(str(self.output[key]))

    def ensure_directories(self) -> None:
        keys = [
            ("models", "apacheone_dir"),
            ("models", "bfl_dir"),
            ("models", "experimental_dir"),
            ("cache", "root"),
            ("cache", "prompt_dir"),
            ("cache", "user_photo_dir"),
            ("cache", "logo_dir"),
            ("cache", "kv_dir"),
            ("output", "root"),
            ("output", "diagnostics"),
        ]
        for section, key in keys:
            raw_section = self.raw.get(section, {})
            if key in raw_section:
                self.resolve_path(str(raw_section[key])).mkdir(parents=True, exist_ok=True)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / CONFIG_RELATIVE_PATH).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {CONFIG_RELATIVE_PATH}. Run from the project root."
    )


def load_project_config(config_path: str | Path | None = None) -> ProjectConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pyyaml. Install requirements first.") from exc

    if config_path is None:
        root = find_project_root()
        path = root / CONFIG_RELATIVE_PATH
    else:
        path = Path(config_path).resolve()
        root = path.parent.parent

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return ProjectConfig(root=root.resolve(), raw=raw)
