"""
app.py — Rising Waters Flask Application with SQLite Database
==============================================================
"""
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import json
import os
from datetime import datetime
from database import get_db_connection, init_db
app = Flask(__name__)
app.secret_key = "rising_waters_secret_key_2024"
from joblib import load
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "floods.save")
SCALER_PATH   = os.path.join(BASE_DIR, "transform.save")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.json")
model = load(MODEL_PATH)
sc = load(SCALER_PATH)
scaler = sc
with open(FEATURES_PATH) as f:
    FEATURE_COLUMNS = json.load(f)
init_db()
def get_risk_level(probability):
    if probability >= 0.75:
        return "Critical", "#d9534f"
    elif probability >= 0.55:
        return "High", "#e67e22"
    elif probability >= 0.35:
        return "Moderate", "#f0ad4e"
    else:
        return "Low", "#28a745"
def is_logged_in():
    return "user_id" in session
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if is_logged_in():
        return redirect(url_for("home"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        role     = request.form.get("role", "Local Authority")
        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("signup.html")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM User WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            flash("Username or email already registered.", "danger")
            conn.close()
            return render_template("signup.html")
        try:
            hashed_pwd = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO User (username, email, password, role) VALUES (?, ?, ?, ?)",
                (username, email, hashed_pwd, role)
            )
            conn.commit()
            flash("Registration successful! Please login.", "success")
            conn.close()
            return redirect(url_for("login"))
        except Exception as e:
            flash(f"Error during registration: {str(e)}", "danger")
            conn.close()
    return render_template("signup.html")
@app.route("/login", methods=["GET", "POST"])
def login():
    if is_logged_in():
        return redirect(url_for("home"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Please enter both username and password.", "danger")
            return render_template("login.html")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM User WHERE username = ? OR email = ?", (username, username))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["user_id"]  = user["user_id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("home"))
        else:
            flash("Invalid username/email or password.", "danger")
    return render_template("login.html")
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))
@app.route("/")
def home():
    if not is_logged_in():
        return render_template("home.html")
    conn = get_db_connection()
    cursor = conn.cursor()
    is_admin_or_officer = session.get("role") in ["Admin", "Disaster Management Officer"]
    if is_admin_or_officer:
        cursor.execute("SELECT COUNT(*) FROM Prediction")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM Prediction WHERE prediction_result = 'Flood'")
        floods = cursor.fetchone()[0]
    else:
        cursor.execute("""
            SELECT COUNT(*) FROM Prediction p
            JOIN WeatherData w ON p.data_id = w.data_id
            WHERE w.user_id = ?
        """, (session["user_id"],))
        total = cursor.fetchone()[0]
        cursor.execute("""
            SELECT COUNT(*) FROM Prediction p
            JOIN WeatherData w ON p.data_id = w.data_id
            WHERE w.user_id = ? AND p.prediction_result = 'Flood'
        """, (session["user_id"],))
        floods = cursor.fetchone()[0]
    no_floods = total - floods
    cursor.execute("SELECT * FROM MachineLearningModel ORDER BY accuracy DESC")
    models_db = cursor.fetchall()
    cursor.execute("SELECT accuracy FROM MachineLearningModel WHERE algorithm_type = 'XGBoost'")
    row_xgb = cursor.fetchone()
    xgb_accuracy = row_xgb["accuracy"] if row_xgb else 90.80
    if is_admin_or_officer:
        cursor.execute("""
            SELECT p.*, w.annual_rainfall, w.cloud_cover 
            FROM Prediction p
            JOIN WeatherData w ON p.data_id = w.data_id
            ORDER BY p.prediction_id DESC LIMIT 5
        """)
    else:
        cursor.execute("""
            SELECT p.*, w.annual_rainfall, w.cloud_cover 
            FROM Prediction p
            JOIN WeatherData w ON p.data_id = w.data_id
            WHERE w.user_id = ?
            ORDER BY p.prediction_id DESC LIMIT 5
        """, (session["user_id"],))
    recent_history = cursor.fetchall()
    history_list = []
    for row in recent_history:
        history_list.append({
            "timestamp": row["prediction_date"],
            "label":     row["prediction_result"],
            "probability": round(row["flood_probability"] * 100, 2),
            "risk":      get_risk_level(row["flood_probability"])[0],
            "rainfall":  row["annual_rainfall"],
            "visibility":row["cloud_cover"]
        })
    conn.close()
    return render_template("dashboard.html",
                           total=total,
                           floods=floods,
                           no_floods=no_floods,
                           xgb_accuracy=xgb_accuracy,
                           models=models_db,
                           history=history_list)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not is_logged_in():
        return redirect(url_for("login"))
    if request.method == "POST":
        try:
            form = request.form
            features = {
                "Temp":        float(form.get("Temp", 29)),
                "Humidity":    float(form.get("Humidity", 70)),
                "Cloud Cover": float(form.get("Cloud Cover", 30)),
                "ANNUAL":      float(form.get("ANNUAL", 3200)),
                "Jan-Feb":     float(form.get("Jan-Feb", 70)),
                "Mar-May":     float(form.get("Mar-May", 380)),
                "Jun-Sep":     float(form.get("Jun-Sep", 2100)),
                "Oct-Dec":     float(form.get("Oct-Dec", 660)),
                "avgjune":     float(form.get("avgjune", 270)),
                "sub":         float(form.get("sub", 650)),
            }
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO WeatherData (
                    user_id, temp, humidity, cloud_cover, annual_rainfall, 
                    jan_feb, mar_may, jun_sep, oct_dec, avgjune, sub
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session["user_id"], features["Temp"], features["Humidity"],
                features["Cloud Cover"], features["ANNUAL"], features["Jan-Feb"],
                features["Mar-May"], features["Jun-Sep"], features["Oct-Dec"],
                features["avgjune"], features["sub"]
            ))
            data_id = cursor.lastrowid
            input_arr = np.array([[features[col] for col in FEATURE_COLUMNS]])
            input_scaled = scaler.transform(input_arr)
            prediction = model.predict(input_scaled)[0]
            try:
                raw_proba = float(model.predict_proba(input_scaled)[0][1])
            except Exception:
                raw_proba = 1.0 if prediction == 1 else 0.0
            annual   = features["ANNUAL"]
            jun_sep  = features["Jun-Sep"]
            mar_may  = features["Mar-May"]
            jan_feb  = features["Jan-Feb"]
            annual_score  = min(annual / 4500.0, 1.0)
            junsep_score  = min(jun_sep / 3500.0, 1.0)
            marmay_score  = min(mar_may / 900.0, 1.0)
            janfeb_score  = min(jan_feb / 100.0, 1.0)
            soft_score    = (
                0.40 * annual_score +
                0.35 * junsep_score +
                0.15 * marmay_score +
                0.10 * janfeb_score
            )
            probability = round(0.60 * raw_proba + 0.40 * soft_score, 4)
            label     = "Flood" if prediction == 1 else "No Flood"
            risk, _   = get_risk_level(probability)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("SELECT model_id FROM MachineLearningModel WHERE algorithm_type = 'XGBoost'")
            row_model = cursor.fetchone()
            model_id = row_model["model_id"] if row_model else 4
            cursor.execute("""
                INSERT INTO Prediction (
                    data_id, model_id, prediction_result, flood_probability, prediction_date
                ) VALUES (?, ?, ?, ?, ?)
            """, (data_id, model_id, label, probability, timestamp))
            conn.commit()
            conn.close()
            session["result"] = {
                "label":       label,
                "probability": round(probability * 100, 2),
                "risk":        risk,
                "timestamp":   timestamp,
                "features":    features,
                "temperature": features["Temp"],
                "humidity":    features["Humidity"],
            }
            return redirect(url_for("result"))
        except Exception as e:
            return render_template("index.html", error=str(e))
    return render_template("index.html")
@app.route("/result")
def result():
    if not is_logged_in():
        return redirect(url_for("login"))
    result_data = session.get("result")
    if not result_data:
        return redirect(url_for("predict"))
    if result_data["label"] == "Flood":
        return render_template("chance.html", **result_data)
    else:
        return render_template("no_chance.html", **result_data)
@app.route("/history")
def history():
    if not is_logged_in():
        return redirect(url_for("login"))
    conn = get_db_connection()
    cursor = conn.cursor()
    is_admin_or_officer = session.get("role") in ["Admin", "Disaster Management Officer"]
    if is_admin_or_officer:
        cursor.execute("""
            SELECT p.*, w.annual_rainfall, w.cloud_cover, w.temp, w.humidity, u.username, u.role
            FROM Prediction p
            JOIN WeatherData w ON p.data_id = w.data_id
            JOIN User u ON w.user_id = u.user_id
            ORDER BY p.prediction_id DESC
        """)
    else:
        cursor.execute("""
            SELECT p.*, w.annual_rainfall, w.cloud_cover, w.temp, w.humidity, u.username, u.role
            FROM Prediction p
            JOIN WeatherData w ON p.data_id = w.data_id
            JOIN User u ON w.user_id = u.user_id
            WHERE w.user_id = ?
            ORDER BY p.prediction_id DESC
        """, (session["user_id"],))
    predictions = cursor.fetchall()
    conn.close()
    history_list = []
    for row in predictions:
        history_list.append({
            "timestamp":  row["prediction_date"],
            "label":      row["prediction_result"],
            "probability": round(row["flood_probability"] * 100, 2),
            "risk":       get_risk_level(row["flood_probability"])[0],
            "rainfall":   row["annual_rainfall"],
            "visibility": row["cloud_cover"],
            "temp":       row["temp"],
            "humidity":   row["humidity"],
            "username":   row["username"],
            "role":       row["role"]
        })
    return render_template("history.html", history=history_list)
@app.route("/clear_history", methods=["POST"])
def clear_history():
    if not is_logged_in():
        return redirect(url_for("login"))
    conn = get_db_connection()
    cursor = conn.cursor()
    if session.get("role") in ["Admin", "Disaster Management Officer"]:
        cursor.execute("DELETE FROM Prediction")
        cursor.execute("DELETE FROM WeatherData")
    else:
        cursor.execute("""
            DELETE FROM Prediction WHERE data_id IN (
                SELECT data_id FROM WeatherData WHERE user_id = ?
            )
        """, (session["user_id"],))
        cursor.execute("DELETE FROM WeatherData WHERE user_id = ?", (session["user_id"],))
    conn.commit()
    conn.close()
    session.pop("result", None)
    flash("Prediction history cleared successfully.", "success")
    return redirect(url_for("history"))
@app.route("/about")
def about():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("about.html")
if __name__ == "__main__":
    app.run(debug=True)
