import json
from datetime import datetime
import joblib
from xgboost import XGBClassifier
from data import load_and_split
import config as cfg
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def save_metadata(model_type, best_params=None, path=cfg.METADATA_PATH):
    """
    Save model metadata (params + timestamp) to JSON.
    """
    metadata = {
        "model_type": model_type,
        "best_params": best_params,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)

def train_and_save_model(model_path=cfg.MODEL_PATH, api_test_path=cfg.TEST_DATA_PATH):
    """
    Train an XGBoost model with predefined hyperparameters and save it.
    Also saves held-out rows for API testing separately.

    Args:
        model_path (str) : Path to save the trained model.
        api_test_path (str) : Path to save the held-out API test data.

    Returns: None
    """

    logger.info("Starting model training...")
    # Load preprocessed train/test split
    X_train, _, y_train, _, api_data = load_and_split()
    logger.info("Data loaded and split.")
    
    # Save API holdout data
    api_data.to_csv(api_test_path, index=False)
    logger.info(f"API test data saved to {api_test_path}")

    # Handle imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # load tuned hyperparams
    params = cfg.XGB_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight

    # instantiate model with loaded params
    model = XGBClassifier(**params)

    # Train model
    logger.info("Fitting xgb model...")
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, model_path)
    logger.info(f"Model saved -> {cfg.MODEL_PATH}")

    # Save metadata
    save_metadata("tuned-xgb", best_params=params)
    logger.info(f"Model metadata saved -> {cfg.METADATA_PATH}")

if __name__ == "__main__":
    train_and_save_model()
