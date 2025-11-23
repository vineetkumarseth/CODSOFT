import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("titanic.csv")

features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
X = df[features]
y = df["Survived"]

imputer_num = SimpleImputer(strategy="mean")
X[["Age", "Fare"]] = imputer_num.fit_transform(X[["Age", "Fare"]])

imputer_cat = SimpleImputer(strategy="most_frequent")
X[["Embarked"]] = imputer_cat.fit_transform(X[["Embarked"]])

label_enc = LabelEncoder()
X["Sex"] = label_enc.fit_transform(X["Sex"])
X["Embarked"] = label_enc.fit_transform(X["Embarked"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

sample_passenger = pd.DataFrame({
    "Pclass": [3],
    "Sex": label_enc.transform(["male"]),
    "Age": [25],
    "Fare": [7.25],
    "SibSp": [0],
    "Parch": [0],
    "Embarked": label_enc.transform(["S"])
})

print("Prediction (1 = survived, 0 = died):", model.predict(sample_passenger))
