import pandas as pd
from app.monitoring.metric_decay import MetricDecayMonitor

# Example: simulated data
y_true = pd.Series([1, 0, 1, 1, 0, 1])
y_pred = pd.Series([1, 0, 0, 1, 0, 1])

monitor = MetricDecayMonitor()

current_metrics = monitor.compute_metrics(y_true, y_pred)

reference_metrics = {
    "f1": 0.85,
    "precision": 0.83,
    "recall": 0.87,
    "accuracy": 0.84,
}

decay = monitor.detect_decay(reference_metrics, current_metrics)

print("METRIC DECAY:")
print(decay)