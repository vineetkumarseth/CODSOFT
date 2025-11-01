# ============================================================
# PROJECT: CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
# ============================================================

# Importing all required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import SMOTE
import joblib

# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================

# Load your dataset (supports both CSV and Excel)
# Example: data = pd.read_excel("creditcard.xlsx")
data = pd.read_csv("creditcard.csv")

print("✅ Dataset loaded successfully!")
print("Shape of dataset:", data.shape)
print("\nFirst 5 rows:\n", data.head())

# ============================================================
# STEP 2: EXPLORE AND CLEAN DATA
# ============================================================

# Checking for null values
print("\nChecking for missing values...")
print(data.isnull().sum())

# Fill missing values if any (optional)
data = data.fillna(0)

# Ensure 'Class' column exists — this is the target variable
if 'Class' not in data.columns:
    raise Exception("❌ Error: The dataset must have a 'Class' column (1 = Fraud, 0 = Genuine).")

# ============================================================
# STEP 3: SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop('Class', axis=1)
y = data['Class']

print("\nNumber of genuine (0) and fraudulent (1) transactions:")
print(y.value_counts())

# ============================================================
# STEP 4: FEATURE SCALING (NORMALIZATION)
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✅ Feature scaling completed.")

# ============================================================
# STEP 5: HANDLE CLASS IMBALANCE USING SMOTE
# ============================================================

print("\nApplying SMOTE (Synthetic Minority Over-sampling Technique)...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("Before SMOTE:\n", y.value_counts())
print("After SMOTE:\n", y_resampled.value_counts())
print("\n✅ Class imbalance handled successfully.")

# ============================================================
# STEP 6: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# ============================================================
# STEP 7: TRAIN THE MODEL (Random Forest)
# ============================================================

print("\nTraining the Random Forest Classifier model...")

model = RandomForestClassifier(
    n_estimators=100,       # number of trees
    random_state=42,        # reproducibility
    n_jobs=-1,              # use all CPU cores
    max_depth=None,         # expand until all leaves are pure
    min_samples_split=2
)

model.fit(X_train, y_train)

print("✅ Model training completed successfully!")

# ============================================================
# STEP 8: MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# STEP 9: EVALUATE THE MODEL PERFORMANCE
# ============================================================

print("\n📊 MODEL PERFORMANCE REPORT 📊")
print("-------------------------------------------------")
print("Accuracy Score : ", accuracy_score(y_test, y_pred))
print("Precision Score: ", precision_score(y_test, y_pred))
print("Recall Score   : ", recall_score(y_test, y_pred))
print("F1 Score       : ", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# STEP 10: SAVE THE TRAINED MODEL
# ============================================================

joblib.dump(model, 'credit_card_fraud_model.pkl')
print("\n💾 Trained model saved as 'credit_card_fraud_model.pkl'")

# ============================================================
# STEP 11: TEST THE MODEL WITH NEW INPUT (OPTIONAL)
# ============================================================

# Example: Predict on one transaction
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)
print("\n🔍 Example Prediction for one transaction:")
print("Predicted Class:", "Fraud" if prediction[0] == 1 else "Genuine")

print("\n🎉 Program completed successfully!")
