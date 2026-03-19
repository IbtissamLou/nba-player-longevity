from app.monitoring.prediction_monitor import PredictionMonitor

monitor = PredictionMonitor("logs/inference_log.jsonl")

df = monitor.load_predictions()
metrics = monitor.compute_metrics(df)

print("PREDICTION MONITORING:")
print(metrics)