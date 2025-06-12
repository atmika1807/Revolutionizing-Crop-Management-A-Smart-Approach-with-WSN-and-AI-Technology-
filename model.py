import pandas as pd
df = pd.read_csv(r"C:\Users\atpt1\New folder\Revolutionizing Crop Management\crop data.csv")
df.info()
df.describe()
df.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Features and target
X = df.drop(columns=["action"])
y = df["action"]

# Categorical and numerical columns
cat_cols = ["crop_type", "soil_type"]
num_cols = ["soil_moisture", "temperature", "humidity", "nutrient_level", "soil_pH"]

# Pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), cat_cols)
], remainder='passthrough')

model_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier())
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(model_pipeline, "crop_model.pkl")
