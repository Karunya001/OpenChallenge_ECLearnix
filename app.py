import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State
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
    try:
        subprocess.run(["python", "scrape_courses.py"], check=True)
    except Exception as e:
        print("❌ Could not scrape courses:", e)
        demo_courses = pd.DataFrame([
            {"title": "Resume Building", "url": "https://www.eclearnix.com/course/resume/", "Cluster": 1},
            {"title": "International Conference", "url": "https://www.eclearnix.com/course/icies2025/", "Cluster": 0},
            {"title": "Crash Course Starter", "url": "https://eclearnix.com/courses/introduction-to-ai", "Cluster": 2}
        ])
        demo_courses.to_csv(COURSE_FILE, index=False)
        print("✅ Using fallback demo courses.")

courses_df = pd.read_csv(COURSE_FILE)

def get_recommendations(cluster_id, n=2):
    """Fetch at least n recommendations for a cluster"""
    cluster_courses = courses_df[courses_df["Cluster"] == cluster_id]
    if cluster_courses.empty:
        # fallback: pick any random courses
        return courses_df.sample(min(n, len(courses_df)), random_state=42).to_dict("records")
    recs = cluster_courses.sample(min(n, len(cluster_courses)), random_state=42)
    return recs.to_dict("records")

# ---------- Store new users ----------
USER_FILE = "users.csv"
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["User_ID", "Password"]).to_csv(USER_FILE, index=False)

users_df = pd.read_csv(USER_FILE)

# ---------- App ----------
app = Dash(__name__)
app.title = "ECLEARNIX Smart Recommendation System"

app.layout = html.Div([
    dcc.Tabs(id="tabs", value="login", children=[
        dcc.Tab(label="Login", value="login"),
        dcc.Tab(label="Signup", value="signup"),
        dcc.Tab(label="Smart Recommendations", value="recommend"),
    ], style={"fontSize": "20px", "fontWeight": "bold"}),
    html.Div(id="tabs-content"),
    dcc.Store(id="session_user")
])

# ---------- Render Tabs ----------
@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value")
)
def render_content(tab):
    if tab == "login":
        return html.Div([
            html.H2("🔑 Login", style={"color": "#013A63"}),
            dcc.Input(id="login_user", type="text", placeholder="User_ID", style={'margin': '10px', 'padding': '8px'}),
            dcc.Input(id="login_pass", type="password", placeholder="Password", style={'margin': '10px', 'padding': '8px'}),
            html.Button("Login", id="login_btn", n_clicks=0,
                        style={'backgroundColor': '#013A63', 'color': 'white', 'padding': '10px 20px', 'borderRadius': '6px'}),
            html.Div(id="login_message", style={'marginTop': '15px', 'fontSize': '18px'})
        ], style={'textAlign': 'center', 'padding': '30px'})

    elif tab == "signup":
        return html.Div([
            html.H2("📝 Signup", style={"color": "#013A63"}),
            dcc.Input(id="signup_user", type="text", placeholder="Choose User_ID", style={'margin': '10px', 'padding': '8px'}),
            dcc.Input(id="signup_pass", type="password", placeholder="Choose Password", style={'margin': '10px', 'padding': '8px'}),
            html.Button("Signup", id="signup_btn", n_clicks=0,
                        style={'backgroundColor': '#2D6A4F', 'color': 'white', 'padding': '10px 20px', 'borderRadius': '6px'}),
            html.Div(id="signup_message", style={'marginTop': '15px', 'fontSize': '18px'})
        ], style={'textAlign': 'center', 'padding': '30px'})

    elif tab == "recommend":
        return html.Div([
            html.H2("🎯 Smart Recommendations", style={"color": "#013A63"}),
            html.Div(id="recommendations_output", style={'marginTop': '30px'})
        ], style={'textAlign': 'center', 'padding': '30px'})

# ---------- Signup Logic ----------
@app.callback(
    Output("signup_message", "children"),
    Input("signup_btn", "n_clicks"),
    State("signup_user", "value"),
    State("signup_pass", "value")
)
def signup_user(n, user_id, password):
    if n > 0:
        if not user_id or not password:
            return "⚠️ Please provide both User_ID and Password."
        global users_df
        if user_id in users_df["User_ID"].astype(str).values:
            return "❌ User already exists. Please login."
        new_row = pd.DataFrame([[user_id, password]], columns=["User_ID", "Password"])
        users_df = pd.concat([users_df, new_row], ignore_index=True)
        users_df.to_csv(USER_FILE, index=False)
        return f"✅ User {user_id} registered successfully! Please go to Login tab."
    return ""

# ---------- Login Logic ----------
@app.callback(
    Output("login_message", "children"),
    Output("session_user", "data"),
    Input("login_btn", "n_clicks"),
    State("login_user", "value"),
    State("login_pass", "value")
)
def login_user(n, user_id, password):
    if n > 0:
        global users_df
        user_row = users_df[users_df["User_ID"].astype(str) == str(user_id)]
        if not user_row.empty:
            stored_pass = user_row["Password"].values[0]
            if password == stored_pass:
                return f"✅ Welcome {user_id}! Go to 'Smart Recommendations' tab.", user_id
            else:
                return "❌ Incorrect password.", None
        else:
            return "❌ User not found. Please Signup.", None
    return "", None

# ---------- Recommendations ----------
@app.callback(
    Output("recommendations_output", "children"),
    Input("session_user", "data")
)
def generate_recommendations(user_id):
    if user_id:
        if user_id in df["User_ID"].astype(str).values:
            user_data = df[df["User_ID"].astype(str) == str(user_id)]
            features = user_data[['Time_Spent_Total_Minutes', 'Feedback_Rating',
                                  'Time_Spent_Total_Minutes']].values
            cluster_id = model.predict(features)[0]
        else:
            cluster_id = np.random.choice([0, 1, 2])  # random for new users

        recs = get_recommendations(cluster_id, n=2)  # at least 2 courses
        if not recs:
            return "⚠️ No recommendations available."

        # Render styled course cards
        cards = []
        for r in recs:
            cards.append(
                html.Div(style={
                    'border': '2px solid #013A63', 'borderRadius': '10px', 'padding': '20px',
                    'margin': '15px', 'display': 'inline-block', 'width': '280px',
                    'boxShadow': '2px 2px 10px rgba(0,0,0,0.2)', 'backgroundColor': '#F8F9FA'
                }, children=[
                    html.H3(r["title"], style={'fontSize': '20px', 'color': '#013A63'}),
                    html.A("Go to Course", href=r["url"], target="_blank",
                           style={'display': 'inline-block', 'marginTop': '15px',
                                  'padding': '8px 16px', 'backgroundColor': '#013A63',
                                  'color': 'white', 'borderRadius': '6px', 'textDecoration': 'none'})
                ])
            )
        return html.Div(cards, style={'textAlign': 'center'})
    return "⚠️ Please login to see recommendations."

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="127.0.0.1", port=8050)
