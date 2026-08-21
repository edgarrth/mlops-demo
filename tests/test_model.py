from src.config import load_config
from src.model import LoanRenewalModel


def test_model_can_train(tiny_df):
    config = load_config()
    trainer = LoanRenewalModel(42)
    train = tiny_df[tiny_df.MES <= 201506]
    validation = tiny_df[tiny_df.MES == 201507]
    candidates = trainer.compare(train, validation, 0.10)
    assert len(candidates) == 3
    assert all(0 <= c.metrics["roc_auc"] <= 1 for c in candidates)
