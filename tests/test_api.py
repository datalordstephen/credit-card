import requests
import pandas as pd
from src.config import TEST_DATA_PATH

URL = "http://127.0.0.1:8000"
TEST_DATA = pd.read_csv(TEST_DATA_PATH)

def test_home_endpoint():
    response = requests.get(f"{URL}/")
    assert response.status_code == 200
    data = response.json()
    assert "model_info" in data

def test_single_prediction_endpoint():
    single_txn = TEST_DATA.iloc[0].to_dict()
    single_txn.pop("Class", None)

    response = requests.post(f"{URL}/predict", json=single_txn)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "fraud_prediction" in data

def test_batch_prediction_endpoint():
    with open(TEST_DATA_PATH, "rb") as f:
        response = requests.post(f"{URL}/predict-batch", files={"file": ("api_test.csv", f, "text/csv")})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "fraud_probability" in data[0]
    assert "fraud_prediction" in data[0]
