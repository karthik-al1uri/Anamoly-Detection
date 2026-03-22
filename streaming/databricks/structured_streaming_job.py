from pyspark.sql import DataFrame, SparkSession

import argparse
import io
from datetime import datetime, timezone
from pathlib import PurePosixPath

import torch
from PIL import Image

from ml.data.mvtec import build_image_transform
from ml.inference.score_frame import compute_reconstruction_error, load_model_checkpoint
from ml.inference.thresholds import classify_error


def build_session() -> SparkSession:
    return SparkSession.builder.appName("cold-start-defect-detector").getOrCreate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Databricks structured streaming anomaly scoring.")
    parser.add_argument("--frames-path", default="/Volumes/workspace/default/cold_start_phase2/frames", help="Volume path containing image frames.")
    parser.add_argument("--checkpoint-path", default="/Volumes/workspace/default/cold_start_phase2/models/bottle_autoencoder.pt", help="Path to the trained checkpoint file.")
    parser.add_argument("--delta-output-path", default="/Volumes/workspace/default/cold_start_phase2/delta/anomalies_stream", help="Delta output path for anomaly events.")
    parser.add_argument("--checkpoint-location", default="/Volumes/workspace/default/cold_start_phase2/checkpoints/anomalies_stream", help="Structured Streaming checkpoint location.")
    parser.add_argument("--file-pattern", default="*.png", help="Glob used to select frame files.")
    parser.add_argument("--device", default="cpu", help="Inference device used by PyTorch.")
    parser.add_argument(
        "--trigger-mode",
        choices=["available-now", "continuous"],
        default="available-now",
        help="Use available-now for bounded validation or continuous for a long-running stream.",
    )
    return parser.parse_args()


def build_microbatch_scorer(
    model: torch.nn.Module,
    image_size: int,
    threshold: float,
    delta_output_path: str,
    category: str | None,
    threshold_strategy: str | None,
    device: str,
):
    transform = build_image_transform(image_size)

    def score_microbatch(batch_df: DataFrame, batch_id: int) -> None:
        rows = batch_df.select("path", "content").collect()
        if not rows:
            return

        results: list[dict[str, object]] = []
        for row in rows:
            image_path = row["path"]
            image = Image.open(io.BytesIO(row["content"])).convert("RGB")
            frame_tensor = transform(image)
            mse_score = compute_reconstruction_error(frame_tensor, model=model, device=device)
            is_anomaly = bool(classify_error(mse_score, threshold))
            results.append(
                {
                    "event_ts": datetime.now(timezone.utc).isoformat(),
                    "batch_id": int(batch_id),
                    "image_path": image_path,
                    "source_label": PurePosixPath(image_path).parent.name,
                    "category": category,
                    "mse_score": float(mse_score),
                    "threshold": float(threshold),
                    "threshold_strategy": threshold_strategy,
                    "is_anomaly": is_anomaly,
                    "status": "anomaly" if is_anomaly else "normal",
                }
            )

        batch_df.sparkSession.createDataFrame(results).write.format("delta").mode("append").save(delta_output_path)

    return score_microbatch


def main() -> None:
    args = parse_args()
    spark = build_session()
    model, metadata = load_model_checkpoint(args.checkpoint_path, map_location=args.device)
    image_size = int(metadata.get("image_size") or 256)
    threshold = metadata.get("threshold")
    if threshold is None:
        raise ValueError("The checkpoint does not contain a calibrated anomaly threshold.")

    stream_df = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("pathGlobFilter", args.file_pattern)
        .option("recursiveFileLookup", "true")
        .load(args.frames_path)
        .select("path", "content")
    )

    writer = (
        stream_df.writeStream.foreachBatch(
            build_microbatch_scorer(
                model=model,
                image_size=image_size,
                threshold=float(threshold),
                delta_output_path=args.delta_output_path,
                category=metadata.get("category"),
                threshold_strategy=metadata.get("threshold_strategy"),
                device=args.device,
            )
        )
        .option("checkpointLocation", args.checkpoint_location)
    )

    query = writer.trigger(availableNow=True).start() if args.trigger_mode == "available-now" else writer.start()
    query.awaitTermination()


if __name__ == "__main__":
    main()
