# backend/models/clustering.py
import joblib
import os
from sklearn.cluster import KMeans

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

class UserClustering:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = None

    def train(self, X):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
        kmeans.fit(X)
        self.model = kmeans
        joblib.dump(kmeans, os.path.join(ARTIFACTS_DIR, "kmeans.joblib"))
        return kmeans

    def load(self, path=None):
        path = path or os.path.join(ARTIFACTS_DIR, "kmeans.joblib")
        self.model = joblib.load(path)
        return self.model

    def predict(self, X):
        if self.model is None:
            self.load()
        return self.model.predict(X)
