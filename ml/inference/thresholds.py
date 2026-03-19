import numpy as np


def estimate_threshold(errors: list[float], percentile: float = 95.0) -> float:
    if not errors:
        raise ValueError("At least one reconstruction error is required to estimate a threshold.")
    return float(np.percentile(np.asarray(errors, dtype=np.float32), percentile))



def classify_error(error: float, threshold: float) -> int:
    return int(error > threshold)



def compute_binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have the same length.")
    if not labels:
        return {
            "count": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    true_positives = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 1)
    true_negatives = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 0)
    false_positives = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 1)
    false_negatives = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 0)

    accuracy = (true_positives + true_negatives) / len(labels)
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "count": len(labels),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
