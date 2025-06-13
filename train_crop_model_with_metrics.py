import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load cleaned dataset
file_path = r"C:\Users\atpt1\New folder\Revolutionizing Crop Management\crop_dataset_cleaned.csv"
df = pd.read_csv(file_path)
df.dropna(inplace=True)

# Show original class distribution
print("📊 Class Distribution (Before Upsampling):")
print(df['action'].value_counts())

# Upsample 'Fertilize' if needed
df_majority = df[df['action'] != 'Fertilize']
df_minority = df[df['action'] == 'Fertilize']

if df_minority.empty:
    print("❌ ERROR: No 'Fertilize' class found.")
    exit()

df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

print("\n✅ After Upsampling:")
print(df_balanced['action'].value_counts())

# Define features and target
X = df_balanced.drop(columns=["action"])
y = df_balanced["action"]

# Preprocessing pipeline
categorical_cols = ["crop_type", "soil_type"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

# Model pipeline
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧱 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, r"C:\Users\atpt1\New folder\Revolutionizing Crop Management\crop_model.pkl")
print("\n✅ Model saved successfully.")
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# After predictions
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)