# backend/models/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = None

    def load_raw(self, path="../data/ECLEARNIX_Hackathon_10K_Dataset.xlsx"):
        df = pd.read_excel(path)
        return df

    def basic_clean(self, df: pd.DataFrame):
        # Example cleaning — adapt with EDA results
        # Fill small number of NA columns conservatively
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        for c in numeric_cols:
            df[c] = df[c].fillna(df[c].median())

        # Convert obvious timestamp columns (if exist)
        for c in df.columns:
            if "date" in c.lower() or "time" in c.lower():
                try:
                    df[c] = pd.to_datetime(df[c])
                except Exception:
                    pass
        return df

    def select_features(self, df: pd.DataFrame):
        # Choose numeric columns as features by default
        numeric = df.select_dtypes(include=["number"]).copy()
        # Remove identifier columns if present
        for col in ["user_id", "id", "index"]:
            if col in numeric.columns:
                numeric = numeric.drop(columns=[col])
        # Save chosen feature columns list
        self.feature_cols = numeric.columns.tolist()
        return numeric

    def fit_transform(self, df: pd.DataFrame):
        df = self.basic_clean(df)
        X = self.select_features(df)
        Xs = self.scaler.fit_transform(X)
        # persist scaler + feature list
        joblib.dump(self.scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
        joblib.dump(self.feature_cols, os.path.join(ARTIFACTS_DIR, "feature_cols.joblib"))
        return Xs, df

    def transform(self, df: pd.DataFrame):
        # assumes feature_cols saved
        if self.feature_cols is None:
            self.feature_cols = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_cols.joblib"))
        X = df[self.feature_cols].copy()
        scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
        Xs = scaler.transform(X)
        return Xs
