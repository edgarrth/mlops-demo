from src.config import load_config
from src.data import LoanData


def test_validate_and_split(tiny_df):
    config = load_config()
    data = LoanData(config)
    result = data.validate(tiny_df)
    assert result["rows"] == 60
    split = data.temporal_split(tiny_df)
    assert len(split.train) == 30
    assert len(split.validation) == 15
    assert len(split.test) == 15
