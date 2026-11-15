# Credit Card Fraud Detection

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions. The project covers raw data cleaning, exploratory data analysis, model experimentation, and production-ready training and inference scripts.

---

## Project Structure

```text
Credit-card-fraud-detection/
├── Data/
│   ├── raw/
│   │   └── credit_card_fraud_messy.csv   # Raw transactional dataset
│   └── processed/
│       └── cleaned.csv                   # Cleaned dataset output
├── Model/
│   └── best_model.pkl                    # Serialized preprocessing + winning model pipeline
├── Notebooks/
│   ├── 01_data_understanding.ipynb       # Data inspection, anomaly detection, and cleaning rules
│   ├── 02_EDA.ipynb                      # Exploratory data analysis and visualizations
│   └── 03_experiments.ipynb              # Model comparison across 4 classifiers
├── src/
│   ├── preprocessing.py                  # Shared data cleaning and feature engineering module
│   ├── train.py                          # Trains the winning model and saves best_model.pkl
│   └── predict.py                        # Loads saved model to score new transactions
└── README.md# Credit-card-fraud-detection
Machine learning pipeline for credit card fraud classification
