import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("C:/Users/Vineet/Downloads/movies.csv", encoding='cp1252')

print("Initial data shape:", df.shape)
print(df.isna().sum())


df = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']]


df = df.dropna(subset=['Rating'])

df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']] = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']].fillna("Unknown")


X = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = df['Rating']


cat_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nModel Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

print("\n--- Sample Predictions ---")
results = pd.DataFrame({"Actual Rating": y_test.values, "Predicted Rating": y_pred})
print(results.head())
