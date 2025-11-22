import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE
import joblib  


df = pd.read_csv("creditcard.csv")
print("Dataset shape:", df.shape)
print(df["Class"].value_counts())


X = df.drop(["Class", "Time"], axis=1)
y = df["Class"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("After SMOTE:", np.bincount(y_train_res))


lr = LogisticRegression(max_iter=500)
lr.fit(X_train_res, y_train_res)


rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42
)
rf.fit(X_train_res, y_train_res)


lr_pred = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]

print("\n=== LOGISTIC REGRESSION ===")
print(classification_report(y_test, lr_pred))
print("ROC-AUC:", roc_auc_score(y_test, lr_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))


rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\n=== RANDOM FOREST ===")
print(classification_report(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# Save models to disk
joblib.dump(lr, 'logistic_regression_model.pkl')
joblib.dump(rf, 'random_forest_model.pkl')
