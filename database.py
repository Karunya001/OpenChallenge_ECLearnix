# backend/database.py
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            dataset_index INTEGER  -- index into df_clean.joblib for mapping
        )
    """)
    conn.commit()
    conn.close()

def add_user(username: str, password: str, dataset_index: int = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, dataset_index) VALUES (?, ?, ?)",
                  (username, password, dataset_index))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, password, dataset_index FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "username": row[1], "password": row[2], "dataset_index": row[3]}
    return None

def validate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    return user["password"] == password

def set_dataset_index(username: str, dataset_index: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET dataset_index=? WHERE username=?", (dataset_index, username))
    conn.commit()
    conn.close()
