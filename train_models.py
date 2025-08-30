# train_models.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import joblib, os

print("🚀 Training Smart Recommendation Engine models...")

# Load dataset
df = pd.read_excel("ECLEARNIX_Hackathon_10K_Dataset.xlsx").dropna()

# Encode categorical variables
le = LabelEncoder()
for col in ['User_Type', 'Department', 'Region', 'Platform_Source', 'Event_Type', 'Event_Mode']:
    df[col] = le.fit_transform(df[col])

# Feature engineering
df['Time_Spent_per_Event'] = df['Time_Spent_Total_Minutes'] / (df['Saved_Event_Count'] + 1)
df['Engagement_Score'] = df['Time_Spent_Total_Minutes'] + (df['Feedback_Rating'] * 100) - df['Days_Since_Last_Activity']

# Select features for clustering
clustering_features = df[['Time_Spent_Total_Minutes', 'Feedback_Rating', 'Engagement_Score']]

# Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(clustering_features)

# Save artifacts
os.makedirs("artifacts", exist_ok=True)
joblib.dump(kmeans, "artifacts/kmeans.joblib")
df.to_csv("artifacts/cleaned_dataset.csv", index=False)

print("✅ Training complete! Models & dataset saved in artifacts/")
