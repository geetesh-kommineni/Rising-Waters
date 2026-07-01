import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rising_waters.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. User table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS User (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    
    # 2. WeatherData table (New 10 parameters from flood dataset.xlsx)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS WeatherData (
            data_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            temp REAL,
            humidity REAL,
            cloud_cover REAL,
            annual_rainfall REAL,
            jan_feb REAL,
            mar_may REAL,
            jun_sep REAL,
            oct_dec REAL,
            avgjune REAL,
            sub REAL,
            FOREIGN KEY (user_id) REFERENCES User (user_id)
        )
    """)
    
    # 3. MachineLearningModel table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS MachineLearningModel (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            algorithm_type TEXT NOT NULL,
            accuracy REAL NOT NULL,
            model_file TEXT NOT NULL
        )
    """)
    
    # 4. Prediction table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Prediction (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            prediction_result TEXT NOT NULL,
            flood_probability REAL NOT NULL,
            prediction_date TEXT NOT NULL,
            FOREIGN KEY (data_id) REFERENCES WeatherData (data_id),
            FOREIGN KEY (model_id) REFERENCES MachineLearningModel (model_id)
        )
    """)
    
    # Populate MachineLearningModel table if empty
    cursor.execute("SELECT COUNT(*) FROM MachineLearningModel")
    if cursor.fetchone()[0] == 0:
        models = [
            ("Decision Tree Classifier", "Decision Tree", 82.80, "floods.save"),
            ("Random Forest Classifier", "Random Forest", 90.00, "floods.save"),
            ("K-Nearest Neighbors Classifier", "K-Nearest Neighbors", 84.60, "floods.save"),
            ("XGBoost Classifier", "XGBoost", 90.80, "floods.save")
        ]
        cursor.executemany("""
            INSERT INTO MachineLearningModel (model_name, algorithm_type, accuracy, model_file)
            VALUES (?, ?, ?, ?)
        """, models)
        
    # 5. Populate default users if empty
    cursor.execute("SELECT COUNT(*) FROM User")
    if cursor.fetchone()[0] == 0:
        from werkzeug.security import generate_password_hash
        admin_pass = generate_password_hash("adminpassword")
        met_pass = generate_password_hash("metpassword")
        users = [
            ("admin", "admin@risingwaters.org", admin_pass, "Admin"),
            ("meteorologist", "met@risingwaters.org", met_pass, "Meteorologist")
        ]
        cursor.executemany("""
            INSERT INTO User (username, email, password, role)
            VALUES (?, ?, ?, ?)
        """, users)
        
        # 6. Seed mock WeatherData & Predictions for meteorologist (user_id = 2)
        mock_weather = [
            (2, 28.0, 75.0, 40.0, 3326.6, 9.3, 275.7, 2403.4, 638.2, 130.3, 256.4),  # High flood risk
            (2, 29.0, 70.0, 30.0, 3248.6, 73.4, 386.2, 2122.8, 666.1, 274.8, 649.9)  # Safe
        ]
        cursor.executemany("""
            INSERT INTO WeatherData (user_id, temp, humidity, cloud_cover, annual_rainfall, jan_feb, mar_may, jun_sep, oct_dec, avgjune, sub)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, mock_weather)
        
        # Seed mock Predictions
        from datetime import datetime, timedelta
        time1 = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        time2 = (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
        
        predictions = [
            (1, 4, "Flood", 0.85, time1),
            (2, 4, "No Flood", 0.12, time2)
        ]
        cursor.executemany("""
            INSERT INTO Prediction (data_id, model_id, prediction_result, flood_probability, prediction_date)
            VALUES (?, ?, ?, ?, ?)
        """, predictions)
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
