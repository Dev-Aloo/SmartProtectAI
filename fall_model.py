import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ========================
# 1️⃣ LOAD DATASET
# ========================
dataset_path = r"C:\Users\Devraj Singh\Desktop\SmartProtectAI\sisfall_data.csv"
print("# Loading dataset...")
df = pd.read_csv(dataset_path)
print(f"# Dataset loaded successfully! Shape: {df.shape}")
print(df.head())

# ========================
# 2️⃣ FEATURES & LABELS
# ========================
X = df.drop('label', axis=1)
y = df['label']

# ========================
# 3️⃣ TRAIN-TEST SPLIT
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"# Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ========================
# 4️⃣ SCALING
# ========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================
# 5️⃣ GRID SEARCH TUNING
# ========================
print("\n# Tuning Random Forest hyperparameters...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced', {0: 2, 1: 1}]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='recall_macro',   # prioritize recall for both classes
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)

print(f"\n✅ Best Parameters Found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# ========================
# 6️⃣ TRAIN & EVALUATE
# ========================
print("\n# Training best Random Forest model...")
best_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\n✅ Train Accuracy: {train_acc*100:.2f}%")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_test))

# ========================
# 7️⃣ VISUALIZATIONS
# ========================
# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred_test),
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Non-Fall', 'Fall'],
    yticklabels=['Non-Fall', 'Fall']
)
plt.title("Confusion Matrix - Tuned Random Forest (Fall Detection)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 5))
sns.barplot(
    x=importances[indices],
    y=np.array(X.columns)[indices],
    palette="viridis"
)
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
