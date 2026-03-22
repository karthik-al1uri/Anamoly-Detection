import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ml.data.mvtec import create_evaluation_dataset, create_training_dataset
from ml.inference.score_frame import compute_batch_reconstruction_errors
from ml.inference.thresholds import calibrate_threshold, classify_error, compute_binary_metrics
from ml.models.autoencoder import ConvAutoencoder
``

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an autoencoder for MVTec anomaly detection.")
    parser.add_argument("--dataset-root", default="data/raw/mvtec_ad", help="Root directory of the MVTec AD dataset.")
    parser.add_argument("--category", required=True, help="Dataset category to train on, for example bottle or capsule.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize value used for input images.")
    parser.add_argument("--threshold-percentile", type=float, default=95.0, help="Percentile of training errors used as anomaly threshold.")
    parser.add_argument(
        "--threshold-strategy",
        choices=["auto", "percentile", "balanced_accuracy", "f1"],
        default="auto",
        help="Strategy used to choose the anomaly threshold.",
    )
    parser.add_argument("--checkpoint-dir", default="artifacts/models", help="Directory where trained checkpoints will be saved.")
    parser.add_argument("--metrics-dir", default="artifacts/training", help="Directory where training metrics will be saved.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    return parser.parse_args()


def train_step(batch: torch.Tensor, model: ConvAutoencoder, optimizer: optim.Optimizer, device: str) -> float:
    criterion = nn.MSELoss()
    model.train()
    frames = batch.to(device)
    optimizer.zero_grad()
    reconstructed = model(frames)
    loss = criterion(reconstructed, frames)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_epoch(
    dataloader: DataLoader[torch.Tensor],
    model: ConvAutoencoder,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    losses: list[float] = []
    for batch in dataloader:
        losses.append(train_step(batch, model, optimizer, device))
    return sum(losses) / max(len(losses), 1)


def collect_training_errors(
    dataloader: DataLoader[torch.Tensor],
    model: ConvAutoencoder,
    device: str,
) -> list[float]:
    errors: list[float] = []
    for batch in dataloader:
        errors.extend(compute_batch_reconstruction_errors(batch, model, device=device))
    return errors


def collect_evaluation_records(
    dataloader: DataLoader[tuple[torch.Tensor, int, str]],
    model: ConvAutoencoder,
    device: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for frames, batch_labels, batch_paths in dataloader:
        errors = compute_batch_reconstruction_errors(frames, model, device=device)
        label_values = batch_labels.tolist()

        for path, label, error in zip(batch_paths, label_values, errors):
            records.append(
                {
                    "image_path": path,
                    "label": int(label),
                    "mse_score": error,
                }
            )

    return records


def evaluate_records(
    records: list[dict[str, object]],
    threshold: float,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    labels: list[int] = []
    predictions: list[int] = []
    evaluated_records: list[dict[str, object]] = []

    for record in records:
        label = int(record["label"])
        error = float(record["mse_score"])
        prediction = classify_error(error, threshold)
        labels.append(label)
        predictions.append(prediction)
        evaluated_records.append(
            {
                **record,
                "prediction": prediction,
            }
        )

    return compute_binary_metrics(labels, predictions), evaluated_records


def ensure_dataset_available(dataset_root: str, category: str, train_count: int, eval_count: int) -> None:
    category_root = Path(dataset_root) / category
    if not category_root.exists():
        raise FileNotFoundError(
            f"Category '{category}' was not found under '{dataset_root}'. Download MVTec AD and place it under that root."
        )
    if train_count == 0:
        raise ValueError(f"No training images were found in '{category_root / 'train' / 'good'}'.")
    if eval_count == 0:
        raise ValueError(f"No evaluation images were found in '{category_root / 'test'}'.")


def main() -> None:
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    training_dataset = create_training_dataset(args.dataset_root, args.category, image_size=args.image_size)
    evaluation_dataset = create_evaluation_dataset(args.dataset_root, args.category, image_size=args.image_size)
    ensure_dataset_available(args.dataset_root, args.category, len(training_dataset), len(evaluation_dataset))

    training_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=args.batch_size, shuffle=False)

    model = ConvAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_epoch(training_loader, model, optimizer, device)
        epoch_losses.append(epoch_loss)
        print({"epoch": epoch, "loss": epoch_loss})

    training_errors = collect_training_errors(training_loader, model, device)
    evaluation_records = collect_evaluation_records(evaluation_loader, model, device)
    evaluation_errors = [float(record["mse_score"]) for record in evaluation_records]
    evaluation_labels = [int(record["label"]) for record in evaluation_records]
    threshold, threshold_details = calibrate_threshold(
        training_errors,
        percentile=args.threshold_percentile,
        evaluation_errors=evaluation_errors,
        evaluation_labels=evaluation_labels,
        strategy=args.threshold_strategy,
    )
    metrics, predictions = evaluate_records(evaluation_records, threshold)

    checkpoint_dir = Path(args.checkpoint_dir)
    metrics_dir = Path(args.metrics_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{args.category}_autoencoder.pt"
    metrics_path = metrics_dir / f"{args.category}_metrics.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "category": args.category,
            "image_size": args.image_size,
            "epochs": args.epochs,
            "threshold": threshold,
            "threshold_percentile": args.threshold_percentile,
            "threshold_strategy": args.threshold_strategy,
            "threshold_details": threshold_details,
        },
        checkpoint_path,
    )

    summary = {
        "category": args.category,
        "dataset_root": args.dataset_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "image_size": args.image_size,
        "threshold_percentile": args.threshold_percentile,
        "threshold_strategy": args.threshold_strategy,
        "threshold": threshold,
        "threshold_details": threshold_details,
        "train_image_count": len(training_dataset),
        "test_image_count": len(evaluation_dataset),
        "epoch_losses": epoch_losses,
        "evaluation": metrics,
        "predictions": predictions,
        "checkpoint_path": str(checkpoint_path),
    }

    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint_path": str(checkpoint_path), "metrics_path": str(metrics_path), "threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
