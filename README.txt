# 💳 Credit Card Fraud Detection System

> ML-based fraud detection system benchmarking Logistic Regression, Random Forest, and Decision Tree classifiers — with SMOTE for class imbalance and a Flask web interface for real-time predictions.

---

## The Problem

Credit card fraud is a rare but costly event. Real-world transaction datasets are severely imbalanced — fraudulent transactions represent a tiny fraction of total records. A naive model that predicts "legitimate" for everything can achieve 99% accuracy while being completely useless in practice.

This project builds an end-to-end fraud detection pipeline that correctly handles class imbalance, benchmarks multiple classifiers, and deploys a working prediction interface.

---

## Key Results

| Metric | Detail |
|---|---|
| **Models benchmarked** | Logistic Regression · Random Forest · Decision Tree |
| **Class imbalance** | Addressed with SMOTE (training set only) |
| **Hyperparameter tuning** | Grid Search + Cross-Validation |
| **Evaluation metrics** | Accuracy · Precision · Recall · F1-Score |
| **Deployment** | Flask web interface for real-time inference |
| **Dataset** | Kaggle credit card transactions (Jan 2019 – Dec 2020) |

---

## Architecture

```
Raw Transaction Data (Kaggle CSV)
           ↓
Exploratory Data Analysis
           ↓
Preprocessing & Feature Engineering
           ↓
SMOTE — Class Imbalance Correction (training set only)
           ↓
Model Training & Benchmarking
  ├── Logistic Regression (probabilistic baseline)
  ├── Decision Tree (interpretable rules)
  └── Random Forest (ensemble, highest accuracy)
           ↓
Grid Search + Cross-Validation (hyperparameter tuning)
           ↓
Evaluation — Precision · Recall · F1-Score · Confusion Matrix
           ↓
Flask Web Interface — Real-time prediction input/output
```

---

## Model Comparison

| Model | Strengths | Trade-off |
|---|---|---|
| **Logistic Regression** | Fast, transparent, probabilistic output | Lower accuracy on non-linear patterns |
| **Decision Tree** | Interpretable rules, easy to explain | Prone to overfitting |
| **Random Forest** | High accuracy, handles high-dimensional data | Less interpretable |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.x |
| ML Framework | Scikit-learn |
| Class Balancing | imbalanced-learn (SMOTE) |
| Data Processing | Pandas · NumPy |
| Tuning | GridSearchCV · Cross-Validation |
| Web Interface | Flask · HTML · CSS · JavaScript |
| Notebook | Jupyter Notebook |
| Version Control | Git |

---

## Dataset

**Simulated Credit Card Transactions** — Jan 2019 to Dec 2020  
Available on Kaggle: [kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

Download the dataset and place the CSV in the `/data` folder before running.

---

## How to Run

### Prerequisites
- Python 3.9+

```bash
git clone https://github.com/Pranuvar/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### Option 1 — Jupyter Notebook (analysis + training)

```bash
pip install -r requirements.txt

# Open and run Final_code.ipynb
jupyter notebook Final_code.ipynb
```

### Option 2 — Flask Web Interface (real-time prediction)

```bash
cd Flask
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in your browser. Enter transaction features and get an instant fraud/legitimate prediction.

---

## Project Structure

```
Credit-Card-Fraud-Detection/
├── Flask/
│   ├── app.py              # Flask application
│   ├── templates/          # HTML frontend
│   ├── static/             # CSS & JavaScript
│   └── requirements.txt
├── Final_code.ipynb        # Full ML pipeline (EDA → training → evaluation)
├── Final_code.pdf          # Notebook export
├── dataset_link.txt        # Kaggle dataset link
└── README.md
```

---

## Why These Design Choices?

**Why SMOTE only on training data?**  
Applying SMOTE to the full dataset before splitting would cause data leakage — synthetic fraudulent samples would appear in the test set, giving an inflated picture of real-world performance. SMOTE was applied strictly after the train/test split.

**Why benchmark three models?**  
Each serves a different purpose. Logistic Regression is the interpretable baseline every stakeholder understands. Decision Trees show explicit decision rules useful for audit and compliance. Random Forest delivers the best predictive accuracy. Comparing all three gives an honest view of the trade-offs.

**Why Flask?**  
Moving from notebook to a deployed interface is the step that separates a data science experiment from a usable product. The Flask app demonstrates the full cycle — from trained model to real-time prediction.

---

## Author

**Praneeth Varma Danthuluri**
MSc Artificial Intelligence — National College of Ireland
[LinkedIn](https://linkedin.com/in/praneeth-varma-danthuluri) · [Portfolio](https://Pranuvar.github.io)
