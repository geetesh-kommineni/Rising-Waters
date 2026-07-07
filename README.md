# Rising Waters: AI-Powered Flood Prediction System

## 🌐 Live Deployment & Demo Links
* **Live Web Application (Render)**: [https://rising-waters-jfki.onrender.com/](https://rising-waters-jfki.onrender.com/)
* **Project Demonstration Video**: [Google Drive Demo Link](https://drive.google.com/file/d/1om0nWuBKjE5KY0h1-I5MBnTP_kFxwXLl/view?usp=drive_link)
* **Demo Access Credentials**:
  * **Admin / Supervisor**: Username: `admin` | Password: `adminpassword`
  * **Meteorologist / Operator**: Username: `meteorologist` | Password: `metpassword`

---

## Project Description:
Rising Waters harnesses Machine Learning to provide intelligent flood risk prediction, offering disaster management authorities and citizens a fast, data-driven forecasting experience. The platform includes a Flood Predictor module, which calculates flood probability based on weather parameters (Temperature, Humidity, Cloud Cover, and rainfall metrics), a Supervisor Dashboard displaying real-time predictions, historical trends, and model comparisons, and an interactive History Log that records past queries for local analysis.

Using a trained Random Forest and XGBoost ensemble (achieving 96.55% accuracy), the platform processes meteorological parameters to deliver reliable risk classifications. Built with Flask and styled with the premium Lumina Hydro design system, the system ensures a seamless user experience. With secure user authentication using scrypt hashing and modular database structures via SQLite, Rising Waters empowers local authorities to make preemptive decisions and coordinate emergency services with confidence.

---

## 🛠️ Technologies Used
* **Backend Framework**: Flask (Python)
* **Machine Learning Libraries**: Scikit-Learn, XGBoost, Pandas, Numpy, Joblib
* **Database**: SQLite3
* **Frontend Design**: HTML5, Vanilla CSS3 (Lumina Hydro Glassmorphism Style), Vanilla JavaScript
* **Load Testing**: Locust

---

## 📁 Project Structure
```
/project-root
├── dataset/                  - Contains the raw training dataset: flood dataset.xlsx
├── model/                    - Contains trained model binary (model.pkl, floods.save), scaler (scaler.pkl, transform.save), and feature columns JSON
├── static/                   - Frontend styles (main.css) and generated performance reports & confusion matrices (reports/)
├── templates/                - Flask HTML blueprints (home.html, dashboard.html, login.html, signup.html, chance.html, no_chance.html, about.html)
├── app.py                    - Principal Flask application runner containing prediction pipelines and web controller routes
├── database.py               - SQLite schema blueprint and auto-seeding script (generates rising_waters.db)
├── train_model.py            - Machine learning model preprocessing, training, and evaluation pipeline
├── requirements.txt          - Python dependency requirements
├── README.md                 - Project setup and run instructions
├── Procfile / runtime.txt    - Deployment descriptors for Render cloud platform
└── locustfile.py             - Script configuration for Locust performance load testing
```

---

## 🚀 How to Run

### Step 1: Install Dependencies
Activate your virtual environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Initialize SQLite Database
Initialize the schema and seed the default user accounts and prediction history:
```bash
python database.py
```

### Step 3: Train the Classifiers (Optional)
Run the training pipeline to compute metrics, export confusion matrices, and serialize the best XGBoost/Random Forest models:
```bash
python train_model.py
```

### Step 4: Run the Application Locally
Launch the Flask development server loop:
```bash
python app.py
```
Open your browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📊 Model Performance
Accuracies evaluated on stratified cross-validation splits:
* **XGBoost Classifier**: **96.55%**
* **Random Forest Classifier**: **96.55%**
* **Decision Tree Classifier**: **96.55%**
* **K-Nearest Neighbors**: **89.66%**

---

## 🔒 Security Features
* **Scrypt Hashing**: User passwords are encrypted using strong password key derivation functions.
* **Role-Based Isolation**: Meteorologist roles are isolated to their own history logs; Supervisors can audit global history records.
