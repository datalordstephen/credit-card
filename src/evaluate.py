import joblib
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from src.data import load_and_split
from src.config import MODEL_PATH, METADATA_PATH, REPORTS_PATH, METRICS_PATH

def load_metadata(path=METADATA_PATH):
    """
    Load metadata about the saved model.

    Returns:
        dict: Metadata dictionary with model_type, best_params, trained_at.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"model_type": "unknown", "best_params": None, "trained_at": None}

def evaluate_model(model_path=MODEL_PATH):
    """
    Evaluate the trained model on the test set and save performance plots.

    Args:
        model_path (str) : Path to the trained model file.

    Returns: None
    """
    # Reload split (ensures consistency) and model
    _, X_test, _, y_test, _ = load_and_split()
    model = joblib.load(model_path)

    # display model metadata
    metadata = load_metadata()
    print(f"EVALUATING MODEL: {metadata['model_type']}")
    print("TRAINED AT:", metadata.get("trained_at"))

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)

    # Metrics
    print("ROC AUC:", roc_auc)
    print("PR AUC:", pr_auc)

    # Ensure reports folder exists
    os.makedirs(REPORTS_PATH, exist_ok=True)

    # save metrics to JSON
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "classification_report": report
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics saved to {METRICS_PATH}")

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.title('ROC Curve')
    plt.show()
    plt.savefig(f"{REPORTS_PATH}/roc_curve.png")
    plt.close()
    

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.plot([1, 0], [0, 1], color="navy", lw=2, linestyle="--")
    plt.title('Precision-Recall Curve')
    plt.savefig(f"{REPORTS_PATH}/pr_curve.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"{REPORTS_PATH}/confusion_matrix.png")
    plt.close()

    print("✅ Evaluation complete. Plots saved to reports folder.")

if __name__ == "__main__":
    evaluate_model()
