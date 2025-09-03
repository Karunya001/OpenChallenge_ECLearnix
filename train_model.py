import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import joblib

# ---------- Load Dataset ----------
df = pd.read_excel("data.xlsx")
df.dropna(inplace=True)

# ---------- Encode categorical columns ----------
le = LabelEncoder()
for col in ['User_Type', 'Department', 'Region', 'Platform_Source', 'Event_Type', 'Event_Mode']:
    df[col] = le.fit_transform(df[col])

# ---------- Feature Engineering ----------
df['Time_Spent_per_Event'] = df['Time_Spent_Total_Minutes'] / (df['Saved_Event_Count'] + 1)
df['Engagement_Score'] = (
    df['Time_Spent_Total_Minutes'] +
    (df['Feedback_Rating'] * 100) -
    df['Days_Since_Last_Activity']
)

# ---------- Select Features for Clustering ----------
features = df[['Time_Spent_Total_Minutes', 'Feedback_Rating', 'Engagement_Score']]

# ---------- Train KMeans ----------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(features)

# ---------- Save Model ----------
joblib.dump(kmeans, "model.pkl")

print("✅ Model trained and saved as model.pkl")
