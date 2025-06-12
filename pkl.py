import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Step 1: Load your data
df = pd.read_csv(r"C:\Users\atpt1\New folder\Revolutionizing Crop Management\crop data.csv")  # replace with your actual file

# Step 2: Split into features and target
X = df.drop(columns=["action"])
y = df["action"]

# Step 3: Define preprocessing (categorical + numeric)
categorical_cols = ["crop_type", "soil_type"]
numerical_cols = ["soil_moisture", "temperature", "humidity", "nutrient_level", "soil_pH"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder="passthrough")

# Step 4: Create pipeline with model
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Step 5: Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
pipeline.fit(X_train, y_train)

# Step 6: Save model
joblib.dump(pipeline, "crop_model.pkl")
print("✅ Model saved as crop_model.pkl")
