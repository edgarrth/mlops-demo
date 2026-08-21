from src.metrics import calculate_metrics


def test_ranking_metrics():
    y = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    scores = [0.1, 0.2, 0.9, 0.3, 0.8, 0.4, 0.2, 0.1, 0.05, 0.01]
    metrics = calculate_metrics(y, scores, 0.20)
    assert metrics["precision_at_10"] == 1.0
    assert metrics["recall_at_10"] == 1.0
