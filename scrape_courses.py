import base64
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import joblib
import os
import subprocess

# ---------- Load Dataset ----------
df = pd.read_excel("data.xlsx")
df.dropna(inplace=True)

# ---------- Add Simulated Passwords ----------
df["Password"] = df["User_ID"].astype(str).apply(lambda x: "pass" + x[-3:])

# ---------- Load Clustering Model ----------
model = joblib.load("model.pkl")

# ---------- Ensure Course Mapping Exists ----------
COURSE_FILE = "eclearnix_courses_clustered.csv"

if not os.path.exists(COURSE_FILE):
    print("⚠️ Course mapping file not found. Running scrape_courses.py ...")
    subprocess.run(["python", "scrape_courses.py"], check=True)

courses_df = pd.read_csv(COURSE_FILE)

def get_recommendations(cluster_id, n=3):
    cluster_courses = courses_df[courses_df["Cluster"] == cluster_id]
    if cluster_courses.empty:
        return []
    recs = cluster_courses.sample(min(n, len(cluster_courses)), random_state=42)
    return recs.to_dict("records")
