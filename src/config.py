# Data paths
RAW_DATA_PATH = "data/creditcard.csv"
TEST_DATA_PATH = "data/api_test.csv"

# Model paths
MODEL_PATH = "models/best_xgb_model.pkl"
METADATA_PATH = "models/model_metadata.json"

# path for reports
REPORTS_PATH = "reports"
METRICS_PATH = "reports/metrics.json"

# Training settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
API_HOLDOUT = 15

# XGBoost best hyperparameters (from RandomizedSearchCV in my notebook)
XGB_PARAMS = {
    "subsample": 0.8,
    "n_estimators": 200,
    "max_depth": 7,
    "learning_rate": 0.1,
    "gamma": 1,
    "colsample_bytree": 1.0,
    "eval_metric": "aucpr",
    "use_label_encoder": False,
    "n_jobs": -1,
    "random_state": RANDOM_STATE
}