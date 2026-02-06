from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from src.utils.config import load_config, project_root


def _serialize_metrics(metrics) -> dict:
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return metrics
    if hasattr(metrics, "results_dict"):
        return metrics.results_dict
    try:
        return dict(metrics)
    except Exception:
        return {k: v for k, v in vars(metrics).items() if not k.startswith("_")}


def train_model(config_path: str) -> Path:
    config = load_config(config_path)
    root = project_root()
    dataset_yaml = root / "data" / "processed" / "yolo" / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing dataset config: {dataset_yaml}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install requirements-train.txt") from exc

    arch = config["model"].get("arch", "yolov8n")
    model = YOLO(f"{arch}.pt")

    results = model.train(
        data=str(dataset_yaml),
        epochs=config["model"].get("epochs", 50),
        imgsz=config["model"].get("imgsz", 640),
        batch=config["model"].get("batch", 16),
        device=config["model"].get("device", "cpu"),
        patience=config["model"].get("patience", 20),
        project=str(root / "reports" / "runs"),
        name=config.get("project", {}).get("name", "train"),
        exist_ok=True,
    )

    trainer = getattr(model, "trainer", None)
    metrics = _serialize_metrics(getattr(trainer, "metrics", None))
    metrics.update(_serialize_metrics(getattr(results, "results_dict", None)))

    best_path = Path(getattr(trainer, "best", "")) if trainer else None
    last_path = Path(getattr(trainer, "last", "")) if trainer else None

    checkpoints_dir = root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if best_path and best_path.exists():
        shutil.copyfile(best_path, checkpoints_dir / "best.pt")
    if last_path and last_path.exists():
        shutil.copyfile(last_path, checkpoints_dir / "last.pt")

    output = {
        "arch": arch,
        "dataset": str(dataset_yaml),
        "metrics": metrics,
        "best_checkpoint": str(best_path) if best_path else "",
        "last_checkpoint": str(last_path) if last_path else "",
        "copied_best": str(checkpoints_dir / "best.pt") if best_path and best_path.exists() else "",
        "copied_last": str(checkpoints_dir / "last.pt") if last_path and last_path.exists() else "",
    }

    reports_dir = root / "reports" / "logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "train_metrics.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO model for pool detection.")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    output_path = train_model(args.config)
    print(f"Saved training metrics to {output_path}")


if __name__ == "__main__":
    main()
