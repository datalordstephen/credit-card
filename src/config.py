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
    'subsample': 0.6,
    'n_estimators': 300,
    'min_child_weight': 1,
    'max_depth': 5,
    'learning_rate': 0.05,
    'gamma': 2,
    'colsample_bytree': 1.0,
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "random_state": RANDOM_STATE
}