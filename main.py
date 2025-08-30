# backend/main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

from database import init_db, add_user, validate_user, get_user, set_dataset_index
from recommender import Recommender

# ------------------- INIT -------------------
app = FastAPI()
BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# DB init
init_db()

# Load recommender (requires artifacts)
ARTS = os.path.join(BASE_DIR, "artifacts")
if os.path.exists(os.path.join(ARTS, "kmeans.joblib")):
    recommender = Recommender()
else:
    recommender = None


# ------------------- ROUTES -------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Landing page with login form"""
    return templates.TemplateResponse("login.html", {"request": request, "msg": ""})


@app.post("/signup", response_class=HTMLResponse)
def signup(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle signup and map new user to dataset index"""
    mapped_idx = None
    if recommender and recommender.df_clean is not None:
        mapped_idx = recommender.map_username_to_userid(username)

    success = add_user(username, password, mapped_idx)
    if not success:
        return templates.TemplateResponse(
            "login.html", {"request": request, "msg": "User already exists."}
        )

    return templates.TemplateResponse(
        "login.html", {"request": request, "msg": "Signup successful. Please login."}
    )


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Validate login and redirect to dashboard"""
    ok = validate_user(username, password)
    if not ok:
        return templates.TemplateResponse(
            "login.html", {"request": request, "msg": "Invalid username/password."}
        )

    return RedirectResponse(url=f"/dashboard?username={username}", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, username: str = None):
    """Dashboard shows personalized recommendations"""
    if username is None:
        return RedirectResponse(url="/", status_code=302)

    # fetch stored dataset_index if any
    user = get_user(username)
    dataset_index = user.get("dataset_index") if user else None

    if recommender:
        # assign dataset_index if missing
        if dataset_index is None and recommender.df_clean is not None:
            dataset_index = recommender.map_username_to_userid(username)
            set_dataset_index(username, dataset_index)

        rec = recommender.recommend_for_username(username)
    else:
        rec = {
            "cluster": -1,
            "recommendations": ["⚠ Model artifacts missing. Run train_models.py"],
            "notification": "No models available."
        }

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "username": username,
        "cluster": rec.get("cluster", -1),
        "recs": rec.get("recommendations", []),
        "note": rec.get("notification", "")
    })


# ------------------- ENTRYPOINT -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
