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
├── requirements.txt        
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

* ### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

* ### Windows

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training (Optional)
If you'd like to train the model before using it: 

### Download training set
* Navigate to [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) on kaggle to download the dataset
* Insert the unzipped file (rename to `creditcard.csv` if it's not named that) into the `data` folder of the project

### Train the model
```bash
python src/train.py
```

### Evaluate the model
```bash
python src/evaluate.py
```

## ▶️ Usage
To start up the API:

```bash
uvicorn api.app:app --reload
```

### 📡 Example API Requests (Python)
* ### Single Prediction:
```python
import requests

url = "http://127.0.0.1:8000/predict"

sample_txn = {
  "V1": -16.5265065691231, "V2": 8.58497179585822, "V3": -18.6498531851945,
  "V4": 9.50559351508723, "V5": -13.7938185270957, "V6": -2.83240429939747,
  "V7": -16.701694296045, "V8": 7.51734390370987, "V9": -8.50705863675898,
  "V10": -14.1101844415457, "V11": 5.29923634963938, "V12": -10.8340064814734,
  "V13": 1.67112025332681, "V14": -9.37385858364976, "V15": 0.360805641631617,
  "V16": -9.89924654080666, "V17": -19.2362923697613, "V18": -8.39855199494575,
  "V19": 3.10173536885404, "V20": -1.51492343527852, "V21": 1.19073869481428,
  "V22": -1.12767000902061, "V23": -2.3585787697881, "V24": 0.673461328987237,
  "V25": -1.4136996745882, "V26": -0.46276236139933, "V27": -2.01857524875161,
  "V28": -1.04280416970881, "log_amount": 5.900417766089615, "hour_of_day": 11,
  "time_of_day": 0
}


response = requests.post(url, json=sample_txn)
print(response.json())

```

* ### Batch Prediction:
```python
import requests

url = "http://127.0.0.1:8000/predict-batch"
files = {"file": open("data/api_test.csv", "rb")}

response = requests.post(url, files=files)
print(response.json()[:5])  # show first 5 predictions

```

### 📈 Results

+ **ROC-AUC ≈ 0.97**
+ ** PR-AUC ≈ 0.87**

Balanced precision and recall on fraud cases

> Reports and plots are saved under `reports/`.

### 🚀 Coming Soon

+ CI/CD integration with deployment to Render
+ Automated retraining pipeline
+ Streamlit/Gradio dashboard for fraud monitoring
+ Experiments with LightGBM and/or CatBoost