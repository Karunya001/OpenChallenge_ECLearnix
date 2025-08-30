# recommender.py
import pandas as pd
import joblib

# Load artifacts
try:
    kmeans = joblib.load("artifacts/kmeans.joblib")
    df = pd.read_csv("artifacts/cleaned_dataset.csv")
    print("✅ Recommender loaded successfully")
except Exception as e:
    print("⚠️ Could not load artifacts:", e)
    kmeans, df = None, None

# Recommendation logic
def recommend_learning_path(user_id: int):
    if df is None or kmeans is None:
        return ["⚠️ Model not available. Please run train_models.py first."]

    if user_id not in df["User_ID"].values:
        return ["⚠️ User ID not found in dataset"]

    user_row = df[df["User_ID"] == user_id].iloc[0]
    cluster = user_row["Cluster"]

    if cluster == 0:  # Power Users
        return ["Advanced Data Science", "AI Hackathon", "Mentorship Program"]
    elif cluster == 1:  # Event Explorers
        return ["Quick LMS: Python Basics", "Event-to-Course Certificate"]
    elif cluster == 2:  # Drop-offs
        return ["Onboarding Tutorial", "10-min Crash Course"]
    else:  # Dormant
        return ["Reactivation Webinar", "Career Path Starter Pack"]

# Utility
def get_user_list():
    if df is None:
        return []
    return df["User_ID"].tolist()
