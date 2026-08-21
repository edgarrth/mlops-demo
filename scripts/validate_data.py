import json
from src.config import load_config
from src.data import LoanData

config = load_config()
data = LoanData(config)
df = data.load()
print(json.dumps(data.validate(df), indent=2))
