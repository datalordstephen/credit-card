# Credit Card Fraud Detection 🚨💳

An end-to-end machine learning system to detect fraudulent credit card transactions.
Built with XGBoost and served via a FastAPI REST API.
Includes reproducible preprocessing, evaluation metrics, and tests.

# 📂 Project Structure
```graphql
credit-card/
├── api/
│   └── app.py              # FastAPI app
├── data/
│   ├── creditcard.csv      # Raw dataset (too large to upload, download and insert)
│   └── api_test.csv        # Holdout processed data for API testing
├── models/
│   ├── best_xgb_model.pkl  # Trained XGBoost model
│   └── metadata.json       # Model metadata
├── notebooks/
│   ├── playground.ipynb       # feat engineering + baseline + hyperparam tuning
├── reports/
│   ├── metrics.json          # Evaluation metrics
│   ├── confusion_matrix.png  # Plots from evaluation
│   ├── pr_curve.png          # Plots from evaluation
│   └── roc_curve.png        # Plots from evaluation
├── src/
│   ├── config.py              # store configs
│   ├── data.py                # funcs to load and preprocess data
│   ├── evaluate.py            # extract model metrics + plots
│   ├── logger.py              # logger config
│   ├── predict.py             # funcs for single and batch predictions
│   └── train.py               # training code for xgb model
├── tests/
│   ├── test_data_format.py     # test data format
│   └── test_api.py             # test api responses
├── pyproject.toml       
├── uv.lock                     # virtual environment     
└── README.md
```

## Features
* Preprocessing
    * Log-transform transaction amounts
    * Time features: hour of day + categorical time buckets
* Modeling
    * XGBoost tuned with `RandomizedSearchCV`
    * Class imbalance handled via `scale_pos_weight`
* Evaluation
    * Metrics: ROC-AUC, PR-AUC, classification report
    * Fraud vs non-fraud probability distribution plots
* API
    * `/predict` → single transaction fraud probability
    * `/predict-batch` → batch predictions from CSV
* Testing
    * Dataset format checks
    * Model + evaluation artifacts
    * API endpoint responses

## ⚙️ Setup (Mac/Linux/Windows)
### 1. Clone repo:

```bash
git clone https://github.com/datalordstephen/credit-card.git
cd credit-card
```

### 2. Create environment & install dependencies:

* Install UV (fast package manager)
```bash
pip install uv
```

* Create environment and install requirements
```bash
uv init
uv sync
```


## Training (Optional)
If you'd like to train the model before using it: 

### Download training set
* Navigate to [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) on kaggle to download the dataset
* Insert the unzipped file (rename to `creditcard.csv` if it's not named that) into the `data` folder of the project

### Train the model
```bash
uv run python src/train.py
```

### Evaluate the model
```bash
uv run python src/evaluate.py
```

## ▶️ Usage
### Option 1: Locally with UV
To start up the API:

```bash
uv run uvicorn api.app:app --reload
```
### Option 2: Run with Docker 🐳
Docker provides a containerized environment that ensures consistency across different systems. To get started with it:

### **Build the Docker Image**
```bash
docker build -t credit-card-fraud-api .
```

### **Run the container:**
```bash
docker run -p 8000:8000 credit-card-fraud-api
```

#### The API will be accessible at http://localhost:8000

#### **Stop the container (after inference) and view logs**
```bash
docker stop fraud-api
docker logs fraud-api
```

### 📡 Example API Requests (Python)
* ### Single Prediction:
```bash
python predict.py
```

* ### Batch Prediction:
```python
import requests

# localhost or hostel model
url = "http://127.0.0.1:8000/predict-batch" | "https://cc-fraud-service.onrender.com/predict-batch"
files = {"file": open("data/api_test.csv", "rb")}

response = requests.post(url, files=files)
print(response.json()[:5])  # show first 5 predictions

```

### 📈 Results

+ **ROC-AUC ≈ 0.97**
+ **PR-AUC ≈ 0.87**

Balanced precision and recall on fraud cases

> Reports and plots are saved under `reports/`.

### 🚀 Coming Soon

+ CI/CD integration with deployment to Render
+ Automated retraining pipeline
+ Streamlit/Gradio dashboard for fraud monitoring
+ Experiments with LightGBM and/or CatBoost