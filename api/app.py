from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import json
import tempfile
from src.predict import predict_single, predict_batch
from src.config import MODEL_PATH, METADATA_PATH
from src.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Load model
logger.info("Loading model into API...")
model = joblib.load(MODEL_PATH)

# Load metadata
try:
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
except FileNotFoundError:
    metadata = {"model_type": "unknown", "trained_at": None, "best_params": None}
    logger.warning("Metadata file not found — using defaults.")

app = FastAPI(title="Credit Card Fraud Detection API")

class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    log_amount: float
    hour_of_day: int
    time_of_day: int


@app.get("/")
def home():
    """
    Root endpoint showing API status and current model metadata.
    """
    logger.info("Root endpoint accessed.")
    return {
        "message": "Credit Card Fraud Detection API is running 🚀",
        "model_info": {
            "type": metadata.get("model_type"),
            "trained_at": metadata.get("trained_at"),
            "best_params": metadata.get("best_params")
        }
    }

@app.get("/health")
def health_chech():
    """
    Health check endpoint.
    """
    logger.info("Health check endpoint accessed.")
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    """
    Predict fraud probability for a single transaction.
    """
    result = predict_single(transaction.model_dump(), model_path=MODEL_PATH)
    return result


@app.post("/predict-batch")
async def predict_batch_file(file: UploadFile = File(...)):
    """
    Predict fraud probability for a batch of transactions from a CSV file.
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    results = predict_batch(tmp_path, model_path=MODEL_PATH)
    return results
