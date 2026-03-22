import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from ml.inference.score_frame import load_model_checkpoint, score_image_path
from ml.inference.thresholds import classify_error
from streaming.simulator.producer import iterate_frames



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local real-time anomaly scoring over image frames.")
    parser.add_argument("--dataset-path", default="data/raw", help="Path containing image frames to stream.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to a trained autoencoder checkpoint.")
    parser.add_argument("--output-path", default="artifacts/stream/anomaly_events.jsonl", help="Path to the JSONL event log.")
    parser.add_argument("--summary-path", default="artifacts/stream/run_summary.json", help="Path to the JSON summary for the completed stream run.")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for the local simulation.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of frames to process. Use 0 for all frames.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold override. Defaults to checkpoint threshold.")
    parser.add_argument("--image-size", type=int, default=256, help="Fallback resize value if not stored in the checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device.")
    return parser.parse_args()



def append_event(output_path: Path, event: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")



def reset_output_file(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")



def write_summary(summary_path: Path, summary: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")



def main() -> None:
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model, metadata = load_model_checkpoint(args.checkpoint_path, map_location=device)

    threshold = args.threshold
    if threshold is None:
        threshold = metadata.get("threshold")
    if threshold is None:
        raise ValueError("No anomaly threshold was provided and no threshold was found in the checkpoint.")

    image_size = int(metadata.get("image_size", args.image_size))
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)
    frame_limit = args.limit if args.limit > 0 else None
    reset_output_file(output_path)

    total_frames = 0
    anomaly_frames = 0
    status_counts = {"normal": 0, "anomaly": 0}
    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "anomaly": 0, "normal": 0})

    for index, frame_path in enumerate(iterate_frames(args.dataset_path, fps=args.fps), start=1):
        source_label = Path(frame_path).parent.name
        mse_score = score_image_path(frame_path, model, image_size=image_size, device=device)
        is_anomaly = bool(classify_error(mse_score, float(threshold)))
        status = "anomaly" if is_anomaly else "normal"
        event = {
            "event_id": index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_path": str(frame_path),
            "source_label": source_label,
            "mse_score": mse_score,
            "threshold": float(threshold),
            "is_anomaly": is_anomaly,
            "status": status,
        }
        append_event(output_path, event)
        print(json.dumps(event))

        total_frames += 1
        anomaly_frames += int(is_anomaly)
        status_counts[status] += 1
        source_counts[source_label]["total"] += 1
        source_counts[source_label][status] += 1

        if frame_limit is not None and index >= frame_limit:
            break

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": args.dataset_path,
        "checkpoint_path": args.checkpoint_path,
        "output_path": str(output_path),
        "category": metadata.get("category"),
        "image_size": image_size,
        "threshold": float(threshold),
        "threshold_strategy": metadata.get("threshold_strategy"),
        "threshold_details": metadata.get("threshold_details"),
        "fps": args.fps,
        "frame_limit": frame_limit,
        "total_frames": total_frames,
        "anomaly_frames": anomaly_frames,
        "normal_frames": status_counts["normal"],
        "anomaly_rate": (anomaly_frames / total_frames) if total_frames else 0.0,
        "source_counts": dict(source_counts),
    }
    write_summary(summary_path, summary)
    print(json.dumps({"summary_path": str(summary_path), "total_frames": total_frames, "anomaly_frames": anomaly_frames}))


if __name__ == "__main__":
    main()
