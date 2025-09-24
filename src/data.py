# necessary imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config as cfg

def load_data():
    """
    Load the raw credit card fraud dataset from CSV.

    Returns: 
    pd.DataFrame: Raw dataset.
    """

    return pd.read_csv(cfg.RAW_DATA_PATH)

def preprocess_data(df: pd.DataFrame):
    """
    Apply preprocessing to the dataset:
      - Log-transform the transaction amount.
      - Extract 'hour_of_day' from 'Time' (0-23).
      - Bucketize 'hour_of_day' into 4 encoded categories (morning/noon/evening/night).

    Args:
        df (pd.DataFrame) : Raw dataframe containing all features.

    Returns:
        pd.DataFrame: Preprocessed dataframe ready for modeling.
    """
    df = df.copy()

    # Log-transform Amount
    df["log_amount"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Amount"])

    # Hour of day from Time (0–23)
    df["hour_of_day"] = (df["Time"] // 3600) % 24
    df["hour_of_day"] = df["hour_of_day"].astype(int)

    # Time buckets: Morning/Noon/Evening/Night → integer encoding
    df["time_of_day"] = pd.cut(
        df["hour_of_day"],
        bins=[0, 6, 12, 18, 24],
        labels=["Night", "Morning", "Noon", "Evening"],
        right=False
    )
    df["time_of_day"] = df["time_of_day"].map(
        {"Morning": 0, "Noon": 1, "Evening": 2, "Night": 3}
    )

    # Drop raw Time
    df = df.drop(columns=["Time"])

    return df

def load_and_split(test_size: float = cfg.TEST_SIZE, api_holdout: int = cfg.API_HOLDOUT, random_state: int = cfg.RANDOM_STATE):
    """
    Load, preprocess, and split the dataset into train/test sets.
    Also hold out a few rows completely for API testing.

    Args:
        test_size (float): Proportion of the dataset for test set.
        api_holdout (int): Number of rows to hold out for API testing (not used in training/validation).
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.
        api_data (pd.DataFrame): Completely held-out rows for API testing.
    """
    df = load_data()
    df = preprocess_data(df)

    # Hold out API test data
    api_data = df.sample(n=api_holdout, random_state=random_state)
    df = df.drop(api_data.index)

    # Features and labels
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratified split to preserve fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, api_data

