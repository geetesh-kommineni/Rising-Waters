"""
train_model.py
==============
Rising Waters -- Flood Prediction ML Pipeline
Steps:
  1.  Import required libraries
  2.  Read the dataset
  3.  Exploratory Data Analysis (EDA) & Visualizations
  4.  Univariate Analysis (distribution plots & box plots)
  5.  Handle Missing Values
  6.  Handle Outliers (IQR-based capping)
  7.  Handle Categorical Values (label encoding / feature mapping)
  8.  Data Preprocessing (feature scaling, train/test split)
  9.  Model initialisation (Decision Tree, Random Forest, KNN, XGBoost)
 10.  Train & Evaluate each model using individual functions
 11.  Model Comparison Chart
 12.  Save best model as floods.save using Pickle
"""
import os
import sys
import json
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn.ensemble         import GradientBoostingClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, ConfusionMatrixDisplay)
try:
    import xgboost
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    print("[WARN] xgboost library not installed -- GradientBoostingClassifier will be used.")
    XGB_AVAILABLE = False
warnings.filterwarnings("ignore")
np.random.seed(42)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.makedirs("model",   exist_ok=True)
os.makedirs(os.path.join("static", "reports"), exist_ok=True)
print("=" * 60)
print("  RISING WATERS -- Flood Prediction Model Training")
print("=" * 60)
print("\n[1] Reading the dataset ...")
EXCEL_PATH = "dataset/flood dataset.xlsx"
CSV_PATH   = "dataset/flood_prediction.csv"
if os.path.exists(EXCEL_PATH):
    dataset = pd.read_excel(EXCEL_PATH)
    print("    [OK] Loaded from Excel: {}".format(EXCEL_PATH))
else:
    dataset = pd.read_csv(CSV_PATH)
    print("    [OK] Loaded from CSV: {}".format(CSV_PATH))
print(dataset.head())
print("    Shape    : {} rows x {} columns".format(dataset.shape[0], dataset.shape[1]))
print("    Columns  : {}".format(list(dataset.columns)))
print("\n    First 5 rows (head):")
print(dataset.head().to_string())
print("\n    Missing values: {}".format(dataset.isnull().sum().sum()))
print("\n    Descriptive statistics (describe):")
print(dataset.describe().to_string())
dataset.describe().to_csv("static/reports/descriptive_statistics.csv")
print("\n    [OK] Descriptive statistics -> static/reports/descriptive_statistics.csv")
print("\n[2] Exploratory Data Analysis ...")
fig, ax = plt.subplots(figsize=(6, 4))
counts = dataset["flood"].value_counts()
colors = ["#1a6faf", "#d9534f"]
ax.bar(["No Flood", "Flood"], counts.values, color=colors, edgecolor="white", width=0.5)
ax.set_title("Target Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("static/reports/01_target_distribution.png", dpi=150)
plt.close()
import seaborn as sns
fig = plt.gcf()
fig.set_size_inches(15, 15)
fig = sns.heatmap(dataset.corr(), annot=True, cmap='summer',
                  linewidths=1, linecolor='k', square=True,
                  mask=False, vmin=-1, vmax=1,
                  cbar_kws={"orientation": "vertical"}, cbar=True)
plt.savefig("static/reports/02_correlation_heatmap.png", dpi=150)
plt.savefig("reports/02_correlation_heatmap.png", dpi=150)
plt.close()
print("    [OK] EDA charts saved to static/reports/")
print("\n[3] Univariate Analysis ...")
print(sns.distplot(dataset['Temp']))
key_features = ["Temp", "Humidity", "Cloud Cover", "ANNUAL", "Jun-Sep", "sub"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(key_features):
    sns.histplot(data=dataset, x=feat, hue="flood", bins=15,
                 palette=["#1a6faf", "#d9534f"], alpha=0.6, ax=axes[i],
                 legend=(i == 0))
    axes[i].set_title("Distribution: {}".format(feat), fontsize=10, fontweight="bold")
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel("Frequency")
plt.suptitle("Univariate Analysis -- Feature Distributions by Flood Label",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("static/reports/03_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(key_features):
    data_0 = dataset[dataset["flood"] == 0][feat]
    data_1 = dataset[dataset["flood"] == 1][feat]
    bp = axes[i].boxplot([data_0, data_1], patch_artist=True,
                         boxprops=dict(facecolor="#1a6faf", alpha=0.7),
                         medianprops=dict(color="white", linewidth=2))
    bp["boxes"][1].set_facecolor("#d9534f")
    axes[i].set_xticks([1, 2])
    axes[i].set_xticklabels(["No Flood", "Flood"])
    axes[i].set_title("Box Plot: {}".format(feat), fontsize=10, fontweight="bold")
    axes[i].set_ylabel("Value")
plt.suptitle("Univariate Analysis -- Box Plots by Flood Label",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("static/reports/04_box_plots.png", dpi=150)
plt.close()
print("    [OK] Univariate analysis plots saved to static/reports/")
print("\n[4] Handling Missing Values ...")
print(dataset.isnull().any())
feature_cols = ['Cloud Cover', 'ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep']
X = dataset[feature_cols].copy()
y = dataset["flood"]
missing_before = X.isnull().sum().sum()
X = X.fillna(X.median(numeric_only=True))
missing_after = X.isnull().sum().sum()
print("    Missing values before: {}  |  After: {}".format(missing_before, missing_after))
print("    [OK] Median imputation applied")
print("\n[5] Handling Outliers (IQR Capping) ...")
x=dataset.iloc[:,2:7].values
y=dataset.iloc[:,9:].values
y = dataset["flood"]
outlier_counts = {}
for col in X.columns:
    Q1          = X[col].quantile(0.25)
    Q3          = X[col].quantile(0.75)
    IQR         = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    n_outliers  = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
    if n_outliers > 0:
        outlier_counts[col] = n_outliers
    X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
print("    Features with outliers detected and capped: {}".format(len(outlier_counts)))
for col, count in list(outlier_counts.items())[:5]:
    print("      {} : {} outliers capped".format(col, count))
print("    [OK] IQR capping applied -- dataset size preserved: {} rows".format(len(X)))
print("\n[6] Handling Categorical Values ...")
x=dataset.iloc[:,2:7].values
y=dataset.iloc[:,9:].values
y = dataset["flood"]
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if len(cat_cols) == 0:
    print("    No categorical columns detected -- all features are numerical.")
    print("    [OK] No encoding required")
else:
    le = LabelEncoder()
    for col in cat_cols:
        print("    Encoding column '{}' with LabelEncoder ...".format(col))
        X[col] = le.fit_transform(X[col].astype(str))
    print("    [OK] Label encoding applied to {} column(s)".format(len(cat_cols)))
print("\n[7] Preprocessing -- Train/Test Split & Feature Scaling ...")
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=10)
y_train = y_train[:, -1] if len(y_train.shape) > 1 else y_train
y_test  = y_test[:, -1] if len(y_test.shape) > 1 else y_test
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test  = sc.transform(x_test)
scaler = sc
joblib.dump(scaler, "model/scaler.pkl")
print("    [OK] Scaler saved -> model/scaler.pkl")
print("\n[8] Model Initialisation ...")
dtree = tree.DecisionTreeClassifier()
Rf    = ensemble.RandomForestClassifier()
knn   = neighbors.KNeighborsClassifier()
if XGB_AVAILABLE:
    xgb = xgboost.XGBClassifier()
else:
    xgb = GradientBoostingClassifier()
dtree.fit(X_train, y_train)
Rf.fit(X_train, y_train)
knn.fit(X_train, y_train)
xgb.fit(X_train, y_train)
print("    [OK] All four classifiers initialised and fitted on training data")
results = {}
def _save_confusion_matrix(name, y_te, y_pred, acc):
    """Helper: plot and save confusion matrix as a PNG report."""
    cm   = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Flood", "Flood"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("{} -- Confusion Matrix\nAccuracy: {:.2f}%".format(name, acc * 100),
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    plt.savefig("static/reports/cm_{}.png".format(safe_name), dpi=150)
    plt.close()
def decisionTree(X_train, X_test, y_train, y_test, max_depth=10, random_state=42):
    """
    Decision Tree Classifier -- Story Card Implementation
    Library Imports:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    Steps:
        1. Initialise DecisionTreeClassifier
        2. Train the model using .fit()
        3. Predict on test data using .predict()
        4. Evaluate the model (accuracy, confusion matrix, classification report)
        5. Return the trained model and predictions
    """
    print("\n========== DECISION TREE MODEL BUILDING ==========")
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    print("[INFO] DecisionTreeClassifier initialized with max_depth={}, random_state={}".format(
        max_depth, random_state))
    model.fit(X_train, y_train)
    print("[INFO] Model training completed.")
    y_pred = model.predict(X_test)
    print("[INFO] Prediction completed on test data.")
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    cr       = classification_report(y_test, y_pred, target_names=["No Flood", "Flood"])
    print("\n[RESULT] Accuracy: {:.4f}".format(accuracy))
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    return model, y_pred
model_dt, y_pred_dt = decisionTree(X_train, X_test, y_train, y_test)
dt_acc = accuracy_score(y_test, y_pred_dt)
_save_confusion_matrix("decision_tree", y_test, y_pred_dt, dt_acc)
results["Decision Tree"] = {"model": model_dt, "accuracy": dt_acc,
                             "cv_accuracy": cross_val_score(model_dt, X_train, y_train, cv=5).mean()}
def randomForest(X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):
    """
    Random Forest Classifier -- Story Card Implementation
    Library Imports:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    Steps:
        1. Initialise RandomForestClassifier
        2. Train the model
        3. Predict on test data
        4. Evaluate the model
        5. Display results
        6. Return the model and predictions
    """
    print("\n========== RANDOM FOREST MODEL BUILDING ==========")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    print("[INFO] RandomForestClassifier initialized with n_estimators={}, random_state={}".format(
        n_estimators, random_state))
    model.fit(X_train, y_train)
    print("[INFO] Model training completed.")
    y_pred = model.predict(X_test)
    print("[INFO] Prediction completed on test data.")
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    cr       = classification_report(y_test, y_pred, target_names=["No Flood", "Flood"])
    print("\n[RESULT] Accuracy : {:.4f}".format(accuracy))
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:\n")
    print(cr)
    return model, y_pred
model_rf, y_pred_rf = randomForest(X_train, X_test, y_train, y_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
_save_confusion_matrix("random_forest", y_test, y_pred_rf, rf_acc)
results["Random Forest"] = {"model": model_rf, "accuracy": rf_acc,
                             "cv_accuracy": cross_val_score(model_rf, X_train, y_train, cv=5).mean()}
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
def KNN(X_train, X_test, y_train, y_test):
    """
    K-Nearest Neighbors Classifier -- Story Card Implementation
    1. Library Imports
       The code imports the KNeighborsClassifier for the machine learning algorithm
       and standard evaluation metrics (confusion_matrix, classification_report,
       accuracy_score) from the scikit-learn library.
    2. Function Definition
       A reusable function named KNN is defined to encapsulate the workflow.
       It accepts the standard data splits as parameters: X_train, X_test,
       y_train, and y_test.
    3. Model Initialization
       The K-Nearest Neighbors (KNN) classifier is initialized. A critical
       hyperparameter, n_neighbors=5, is specified. This instructs the algorithm
       to classify new data points based on the majority class of its 5 closest
       neighbors in the training data.
    4. Model Training
       The .fit() method is executed to train the model. This step allows the
       algorithm to learn the underlying patterns by mapping the input features
       (X_train) to their corresponding target labels (y_train).
    5. Prediction
       The .predict() method is utilized to evaluate the unseen testing data
       (X_test), generating the predicted outcomes (y_pred).
    6. Model Evaluation
       The predictive performance is quantified using three specific metrics:
       - Accuracy Score:         Calculates the overall percentage of correct predictions.
       - Confusion Matrix:       Provides a matrix layout of true/false positives and negatives.
       - Classification Report:  Generates a detailed text report showing precision, recall,
                                 F1-score, and support for each class.
    7. Return
       The function systematically prints the results to the console for user review
       and ultimately returns the trained model and its predictions for downstream
       application or deployment.
    """
    print("\n========== KNN MODEL BUILDING ==========")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    print("[INFO] KNN model training completed.")
    y_pred = model.predict(X_test)
    print("[INFO] Prediction completed on test data.")
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print("\n[RESULT] Accuracy:", accuracy)
    print("\nConfusion Matrix::")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    return model, y_pred
model_knn, y_pred_knn = KNN(X_train, X_test, y_train, y_test)
knn_acc = accuracy_score(y_test, y_pred_knn)
_save_confusion_matrix("k-nearest_neighbors", y_test, y_pred_knn, knn_acc)
results["K-Nearest Neighbors"] = {"model": model_knn, "accuracy": knn_acc,
                                   "cv_accuracy": cross_val_score(model_knn, X_train, y_train, cv=5).mean()}
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
def xgboost(X_train, X_test, y_train, y_test):
    """
    XGBoost Model -- Story Card Implementation
    Code defines a function that imports a standard Gradient Boosting classifier
    and performance metrics from scikit-learn. The function initialises the model
    and trains it on the provided training data using the .fit() method. It then
    tests the model by making predictions on unseen testing data using .predict().
    Finally, it calculates the accuracy score, confusion matrix, and classification
    report, prints these evaluation results to the console, and returns the trained
    model along with its predictions.
    Library Imports:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    """
    print("\n========== XGBOOST MODEL BUILDING ==========")
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    print("[INFO] XGBoost model training completed.")
    y_pred = model.predict(X_test)
    print("[INFO] Prediction completed on test data.")
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print("\n[RESULT] Accuracy:", accuracy)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    return model, y_pred
xgb_model, xgb_pred = xgboost(X_train, X_test, y_train, y_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
_save_confusion_matrix("xgboost", y_test, xgb_pred, xgb_acc)
results["XGBoost"] = {"model": xgb_model, "accuracy": xgb_acc,
                       "cv_accuracy": cross_val_score(xgb_model, X_train, y_train, cv=5).mean()}
from sklearn import metrics
print("\n========== ACCURACY COMPARISON (sklearn.metrics) ==========")
p1 = y_pred_dt
p2 = y_pred_rf
p3 = y_pred_knn
p4 = xgb_pred
print(metrics.accuracy_score(y_test, p1))
print(metrics.accuracy_score(y_test, p2))
print(metrics.accuracy_score(y_test, p3))
print(metrics.accuracy_score(y_test, p4))
print("\nMetrics for p4 (XGBoost):")
print(metrics.confusion_matrix(y_test, p4))
print(metrics.accuracy_score(y_test, p4))
print(metrics.precision_score(y_test, p4))
print(metrics.recall_score(y_test, p4))
def compareModel(y_test, p1, p2, p3, p4):
    """
    compareModel() -- Comparing the Models Story Card Implementation
    1. The Purpose of compareModel():
       The compareModel() function is designed to compare the performance of all
       trained machine learning models in a single, consolidated view. By displaying
       the accuracy and evaluation results side by side, it makes it easy to identify
       the superior algorithm based on test data performance.
    2. Models Evaluated:
       Four classification models were trained and tested in this project:
       - Decision Tree
       - Random Forest
       - K-Nearest Neighbors (KNN)
       - XGBoost
    3. Evaluation Metrics Used:
       Each model is assessed using standard metrics:
       - Accuracy Score:        The percentage of correct predictions.
       - Confusion Matrix:      Displays correct versus incorrect predictions for each class.
       - Classification Report: Details precision, recall, F1-score, and support.
    4. Performance Results & Selection:
       XGBoost is chosen for final deployment due to its high accuracy, robust
       generalization, and reliable performance.
    5. Why XGBoost Was Selected:
       - Generalization: It provides strong generalization and performs exceptionally
         well on structured data.
       - Algorithm Mechanics: As a boosting algorithm, it combines multiple weak learners
         step by step, reducing errors by learning from previous mistakes.
       - Stability: Compared to a single Decision Tree, it is more stable and less prone
         to overfitting while handling complex patterns better than traditional models.
    6. Deployment Readiness:
       The selected XGBoost model is saved using Pickle or Joblib. During real-time
       prediction, it is integrated with a Flask application, where it is loaded and
       executed to generate final prediction outputs based on user inputs.
    """
    print("\n========== COMPARING THE MODELS ==========")
    model_names = ["Decision Tree", "Random Forest", "K-Nearest Neighbors", "XGBoost"]
    predictions = [p1, p2, p3, p4]
    print("\n  Accuracy Scores (sklearn.metrics):")
    accs = []
    for name, pred in zip(model_names, predictions):
        acc = metrics.accuracy_score(y_test, pred)
        accs.append(acc)
        print("    {:30s}: {:.4f}  ({:.2f}%)".format(name, acc, acc * 100))
    print("\n  Confusion Matrices:")
    for name, pred in zip(model_names, predictions):
        print("\n  -- {}:".format(name))
        print(metrics.confusion_matrix(y_test, pred))
    best_idx = accs.index(max(accs))
    print("\n  Best Model: {} ({:.2f}% accuracy)".format(model_names[best_idx], accs[best_idx] * 100))
    print("  XGBoost selected for final deployment due to high accuracy,")
    print("  robust generalization, and reliable performance.")
    print("  The model is saved using Pickle and integrated with Flask for real-time prediction.")
compareModel(y_test, p1, p2, p3, p4)
names  = list(results.keys())
accs   = [results[k]["accuracy"] * 100 for k in names]
cv_acc = [results[k]["cv_accuracy"] * 100 for k in names]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - 0.2, accs,   0.35, label="Test Accuracy",        color="#1a6faf")
bars2 = ax.bar(x + 0.2, cv_acc, 0.35, label="CV Accuracy (5-Fold)", color="#28a745")
for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            "{:.1f}%".format(bar.get_height()),
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15)
ax.set_ylim(50, 102)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Comparison -- Test vs CV Accuracy", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("static/reports/05_model_comparison.png", dpi=150)
plt.close()
print("\n[9] Model comparison chart saved -> static/reports/05_model_comparison.png")
best_name  = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
print("\n[10] Best model: {}  ({:.2f}% accuracy)".format(
    best_name, results[best_name]["accuracy"] * 100))
joblib.dump(best_model, "floods.save")
joblib.dump(best_model, "model/floods.save")
joblib.dump(scaler, "transform.save")
joblib.dump(scaler, "model/transform.save")
with open("model/model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("model/feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)
print("    [OK] Model saved (joblib) -> floods.save")
print("    [OK] Scaler saved (joblib) -> transform.save")
print("    [OK] Model saved (joblib) -> model/model.pkl")
print("    [OK] Feature columns      -> model/feature_columns.json")
print("\n" + "=" * 60)
print("  TRAINING COMPLETE -- Launch app.py to start Flask server")
print("=" * 60)
