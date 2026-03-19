def fetch_recent_anomalies() -> list[dict[str, object]]:
    return [
        {
            "timestamp": "2026-03-19T00:00:00Z",
            "mse_score": 0.084,
            "image_path": "s3://placeholder-bucket/frame_0001.png",
            "status": "anomaly",
        }
    ]
