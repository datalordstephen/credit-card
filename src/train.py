import json
from datetime import datetime
import joblib
from xgboost import XGBClassifier
from data import load_and_split
import config as cfg
from logger import get_logger
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    PrecisionRecallDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay
)

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
    
    return metadata

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
    X_train, y_train, X_test, y_test, api_data = load_and_split()
    logger.info("Data loaded and split.")

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
    logger.info(f"Model saved -> {model_path}")

    # Save metadata
    metadata = save_metadata("tuned-xgb", best_params=params)
    logger.info(f"Model metadata saved -> {cfg.METADATA_PATH}")

    return metadata, model, (X_test, y_test)

def evaluate_model(metadata, model, data):
    """
    Evaluate the trained model on the test set and save performance plots.

    Args:
        metadata (dict) : Model metadata dictionary.
        model (XGBClassifier) : Trained model instance.

    Returns: None
    """
    # Load test data and model
    X_test, y_test = data
    logger.info("EVALUATING MODEL...")
    print('-' * 70)
    logger.info("Test Data Loaded.")

    # display model metadata
    logger.info(f"Evaluating model -> {metadata['model_type']} trained at {metadata['trained_at']}")

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    logger.info("Computing evaluation metrics...")  
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Metrics
    logger.info(f"PR AUC: {pr_auc:.4f}")

    # Ensure reports folder exists
    os.makedirs(cfg.REPORTS_PATH, exist_ok=True)

    # save metrics to JSON
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall
    }
    
    with open(cfg.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {cfg.METRICS_PATH}")

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.plot([1, 0], [0, 1], color="navy", lw=2, linestyle="--")
    plt.title('Precision-Recall Curve')
    plt.savefig(f"{cfg.REPORTS_PATH}/pr_curve.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(f"{cfg.REPORTS_PATH}/confusion_matrix.png")
    plt.close()

    logger.info("Evaluation complete. Plots saved to reports folder.")


if __name__ == "__main__":
    metadata, model, data = train_and_save_model()
    evaluate_model(metadata, model, data)
