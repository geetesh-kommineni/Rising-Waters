# Rising Waters 
## A Machine Learning Approach to Flood Prediction

An intelligent flood risk prediction system that uses machine learning to analyse meteorological and environmental data — helping disaster management authorities make timely, data-driven decisions.

---

## 🌟 Key Features & Enhancements

* **Regional Rainfall Modeling**: Re-trained on regional datasets utilizing 5 independent features (`Cloud Cover`, `ANNUAL`, `Jan-Feb`, `Mar-May`, `Jun-Sep`) to classify flood risk (`flood`).
* **Stitch Design System (Lumina Hydro)**: Fully styled using a clean light theme with frosted-glass containers, radial mesh gradients, and a dynamic Light/Dark Mode switcher.
* **Hashed User Authentication**: Secure login and signup systems utilizing `werkzeug.security` (scrypt password hashing).
* **Database Seeding**: Pre-loaded test accounts (`admin` / `adminpassword` and `meteorologist` / `metpassword`) and mock prediction histories to test the system immediately out-of-the-box.
* **Client-side Form Validation**: Real-time checking of parameters (temperature, humidity, rainfall boundaries) inside `static/main.js` with inline errors.

---

## 🛠️ Project Structure

```
Rising waters/
├── app.py                      # Flask web application
├── database.py                 # SQLite schema initialization and seeding
├── train_model.py              # ML pipeline (EDA, preprocessing, training)
├── requirements.txt            # Python dependencies
├── Procfile                    # Deployment config
├── runtime.txt                 # Python version config
├── .gitignore                  # Git ignore definitions
├── dataset/
│   └── flood dataset.xlsx      # Meteorological training dataset
├── model/
│   ├── floods.save             # Saved best classifier (Joblib)
│   ├── transform.save          # Saved scaler (Joblib)
│   ├── model.pkl               # Pickle serialized model
│   └── feature_columns.json    # JSON list of independent features
├── reports/                    # Generated EDA plots & training confusion matrices
│   ├── 02_correlation_heatmap.png
│   ├── 03_feature_distributions.png
│   ├── 05_model_comparison.png
│   ├── cm_xgboost.png
│   ├── cm_random_forest.png
│   └── cm_decision_tree.png
├── templates/
│   ├── home.html               # Landing page
│   ├── index.html              # Weather parameters input form
│   ├── dashboard.html          # Supervisor statistics dashboard
│   ├── history.html            # Prediction logs screen
│   ├── chance.html             # Flood warning result alert page
│   ├── no_chance.html          # Safe / No-flood result page
│   ├── about.html              # About & Tech stack page
│   ├── login.html              # User login
│   └── signup.html             # User registration
└── static/
    ├── css/
    │   └── style.css           # Base modern style rules
    ├── main.css                # Premium overrides & theme selectors
    └── main.js                 # Theme switcher & form validation
```

---

## 🚀 Quick Start

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Initialize Database and Seed Accounts
```bash
python database.py
```
This automatically sets up the SQLite schema and creates the following test credentials:
* **Admin / Supervisor**: `admin` / `adminpassword`
* **Meteorologist**: `meteorologist` / `metpassword`

### Step 3 — Train the Models (Optional)
```bash
python train_model.py
```
Running this script executes:
* Exploratory Data Analysis (generates heatmap, distributions, box plots in `reports/`)
* Preprocessing (median imputation, outlier IQR capping, training/test splits)
* Training of Decision Tree, Random Forest, KNN, and XGBoost classifiers
* Saves the serialized model artifacts to `model/`

### Step 4 — Run the Web App
```bash
python app.py
```
Open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📊 Model Performance

Evaluated accuracies using stratified cross-validation on test splits:

| Model | Accuracy | Serialized Artifact |
| :--- | :--- | :--- |
| **XGBoost** | **95.65%** | `model.pkl` / `floods.save` |
| **Random Forest** | **95.65%** | `floods.save` |
| **Decision Tree** | **95.65%** | `floods.save` |
| **K-Nearest Neighbors** | **91.30%** | `floods.save` |

---

## 🔒 Security & Data Privacy

* **Hashed Storage**: Passwords are never stored in plaintext. They are encrypted using `scrypt` key derivation functions.
* **Role-Based Isolation**: 
  * Standard roles (Meteorologist, Local Authority) only see their own submitted prediction histories.
  * Supervisor roles (Admin, Disaster Management Officer) see global aggregate statistics, and are authorized to clear history records.
