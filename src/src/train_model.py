# src/train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_data(processed_csv: str):
    df = pd.read_csv(processed_csv, parse_dates=['date'])
    return df

def prepare_features(df: pd.DataFrame):
    # choose features
    features = ['product','quantity','total_spent','total_quantity','month','dayofweek','recency_days']
    X = df[features].copy()
    y = df['label_repeat_30d'].astype(int)

    # One-hot encode product
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    prod_ohe = enc.fit_transform(X[['product']])
    prod_cols = enc.get_feature_names_out(['product'])

    # numeric features
    numeric = X[['quantity','total_spent','total_quantity','month','dayofweek','recency_days']].fillna(0).values

    X_final = np.hstack([prod_ohe, numeric])
    return X_final, y.values, enc

def train_and_save(processed_csv: str, model_dir: str = "../models"):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    df = load_data(processed_csv)
    X, y, enc = prepare_features(df)

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # simple RandomForest; you can tune with GridSearchCV
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("RandomForest Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Save model and encoder
    with open(Path(model_dir)/"rf_model.pkl","wb") as f:
        pickle.dump(rf, f)
    with open(Path(model_dir)/"encoder.pkl","wb") as f:
        pickle.dump(enc, f)

    print(f"Saved model and encoder to {model_dir}")

if __name__ == "__main__":
    train_and_save("../data/processed/processed_purchases.csv", "../models")
