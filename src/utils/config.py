from pathlib import Path
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
