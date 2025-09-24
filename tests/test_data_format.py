import pandas as pd
from src.config import TEST_DATA_PATH

def test_data_columns():
    """
    Ensure the dataset contains all expected columns after preprocessing (using api_test.csv).
    """
    df = pd.read_csv(TEST_DATA_PATH)
    expected_cols = {f"V{i}" for i in range(1, 29)} | {"Class", "log_amount", "hour_of_day", "time_of_day"}
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

def test_class_distribution():
    """Ensure target column has both fraud (1) and non-fraud (0) cases. (using api_test.csv)"""
    df = pd.read_csv(TEST_DATA_PATH)
    assert df["Class"].nunique() == 2, "Target column must contain 0 and 1"
