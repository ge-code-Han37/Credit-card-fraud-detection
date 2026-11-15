import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from preprocessing import clean_data

def train_final_model():
    
    raw_path = '../Data/raw/credit_card_fraud_messy.csv'
    if not os.path.exists(raw_path):
        raw_path = 'Data/raw/credit_card_fraud_messy.csv'
        
    df = pd.read_csv(raw_path)
    df_clean = clean_data(df, drop_duplicates=True)
    
    
    target = 'is_fraud'
    drop_cols = ['transaction_id', 'customer_id', 'transaction_datetime', target]
    feature_cols = [c for c in df_clean.columns if c not in drop_cols]
    
    X = df_clean[feature_cols]
    y = df_clean[target]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Final Pipeline with Random Forest (The Best Model)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("Training final Random Forest model...")
    pipeline.fit(X_train, y_train)
    
    
    model_dir = '../Model' if os.path.exists('../Data') else 'Model'
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to {model_path}")
if __name__ == '__main__':
    train_final_model()