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
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
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
    specificity = true_negatives / max(true_negatives + false_positives, 1)
    balanced_accuracy = (recall + specificity) / 2
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "count": len(labels),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }



def optimize_threshold(
    labels: list[int],
    errors: list[float],
    metric: str = "balanced_accuracy",
) -> tuple[float, dict[str, float | int]]:
    if len(labels) != len(errors):
        raise ValueError("Labels and errors must have the same length.")
    if not errors:
        raise ValueError("At least one reconstruction error is required to optimize a threshold.")
    if metric not in {"balanced_accuracy", "f1"}:
        raise ValueError("Supported optimization metrics are 'balanced_accuracy' and 'f1'.")

    unique_errors = np.unique(np.asarray(errors, dtype=np.float32))
    epsilon = max(float(np.finfo(np.float32).eps), 1e-8)
    candidate_thresholds = [float(unique_errors[0] - epsilon)]

    if len(unique_errors) > 1:
        midpoints = (unique_errors[:-1] + unique_errors[1:]) / 2.0
        candidate_thresholds.extend(float(value) for value in midpoints.tolist())

    candidate_thresholds.append(float(unique_errors[-1] + epsilon))

    best_threshold = candidate_thresholds[0]
    best_predictions = [classify_error(error, best_threshold) for error in errors]
    best_metrics = compute_binary_metrics(labels, best_predictions)
    best_key = (
        float(best_metrics[metric]),
        float(best_metrics["f1"]),
        float(best_metrics["precision"]),
        float(best_metrics["recall"]),
    )

    for threshold in candidate_thresholds[1:]:
        predictions = [classify_error(error, threshold) for error in errors]
        metrics = compute_binary_metrics(labels, predictions)
        candidate_key = (
            float(metrics[metric]),
            float(metrics["f1"]),
            float(metrics["precision"]),
            float(metrics["recall"]),
        )
        if candidate_key > best_key:
            best_threshold = threshold
            best_metrics = metrics
            best_key = candidate_key

    return float(best_threshold), best_metrics



def calibrate_threshold(
    training_errors: list[float],
    percentile: float = 95.0,
    evaluation_errors: list[float] | None = None,
    evaluation_labels: list[int] | None = None,
    strategy: str = "auto",
) -> tuple[float, dict[str, object]]:
    percentile_threshold = estimate_threshold(training_errors, percentile)

    if strategy == "percentile":
        return percentile_threshold, {
            "strategy_used": "percentile",
            "threshold_percentile": percentile,
        }

    has_evaluation_data = (
        evaluation_errors is not None
        and evaluation_labels is not None
        and len(evaluation_errors) == len(evaluation_labels)
        and len(evaluation_errors) > 0
    )

    if not has_evaluation_data:
        return percentile_threshold, {
            "strategy_used": "percentile",
            "threshold_percentile": percentile,
            "fallback_reason": "evaluation_data_unavailable",
        }

    optimization_metric = "balanced_accuracy" if strategy == "auto" else strategy
    optimized_threshold, optimized_metrics = optimize_threshold(
        evaluation_labels,
        evaluation_errors,
        metric=optimization_metric,
    )
    return optimized_threshold, {
        "strategy_used": optimization_metric,
        "threshold_percentile": percentile,
        "optimized_metric": optimization_metric,
        "optimized_metric_value": float(optimized_metrics[optimization_metric]),
        "evaluation_metrics": optimized_metrics,
        "percentile_threshold": percentile_threshold,
    }
