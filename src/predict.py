import joblib
import pandas as pd
from config import MODEL_PATH

def load_model(model_path=MODEL_PATH):
    """
    Load a trained model from disk.

    Args:
        model_path (str) : Path to the trained model file.

    Returns:
        model (XGBClassifier) : Loaded model.
    """
    return joblib.load(model_path)

def predict_single(input_data: dict, model_path=MODEL_PATH):
    """
    Make a fraud prediction for a single transaction.

    Args:
        input_data (dict) : Dictionary of features (keys must match training features).
        model_path (str) : Path to the trained model file.

    Returns: 
        dict : Fraud probability and predicted class.
    """
    model = load_model(model_path)
    df = pd.DataFrame([input_data])
    prob = model.predict_proba(df)[:, 1][0]
    pred = int(prob >= 0.5)

    return {"fraud_probability": float(prob), "fraud_prediction": pred}

def predict_batch(file_path: str, model_path=MODEL_PATH):
    """
    Make fraud predictions for a batch of transactions from a CSV file.

    Args:
        file_path (str) : Path to the CSV file containing transaction rows.
        model_path (str) : Path to the trained model file.

    Returns:
        pd.DataFrame : Dataframe with fraud predictions and probabilities.
    """
    model = load_model(model_path)
    df = pd.read_csv(file_path)

    # Drop label if present
    features = df.drop(columns=["Class"], errors="ignore")

    probs = model.predict_proba(features)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df["fraud_probability"] = probs
    df["fraud_prediction"] = preds

    return df
