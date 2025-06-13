import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
import joblib

# Load cleaned dataset
file_path = r"C:\Users\atpt1\New folder\Revolutionizing Crop Management\crop_dataset_cleaned.csv"
df = pd.read_csv(file_path)
df.dropna(inplace=True)

print("🔍 Class Distribution (Before Upsampling):")
print(df['action'].value_counts())

# Upsample minority classes
df_majority = df[df['action'] != 'Fertilize']
df_minority = df[df['action'] == 'Fertilize']

if df_minority.empty:
    print("❌ ERROR: No 'Fertilize' class found. Aborting training.")
    exit()

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
print("\n✅ Class Distribution (After Upsampling):")
print(df_balanced['action'].value_counts())

# Split into X and y
X = df_balanced.drop(columns=["action"])
y = df_balanced["action"]

# Define column types
categorical_cols = ["crop_type", "soil_type"]
numerical_cols = ["soil_moisture", "temperature", "humidity", "nutrient_level", "soil_pH"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

# Full pipeline with classifier
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
pipeline.fit(X_train, y_train)

# Print classes to confirm 'Fertilize' is learned
print("\n🎯 Classes in Trained Model:", pipeline.named_steps['model'].classes_)

# Test with a low nutrient example
test_sample = pd.DataFrame([{
    "crop_type": "Wheat",
    "soil_type": "Clay",
    "soil_moisture": 45,
    "temperature": 28,
    "humidity": 60,
    "nutrient_level": 1,
    "soil_pH": 6.2
}])

prediction = pipeline.predict(test_sample)[0]
proba = pipeline.predict_proba(test_sample)[0]

print("\n🧪 Test Sample Prediction:", prediction)
print("🔎 Prediction Probabilities:")
for cls, p in zip(pipeline.named_steps['model'].classes_, proba):
    print(f"   {cls}: {p:.2f}")

# Save the model
model_path = r"C:\Users\atpt1\New folder\Revolutionizing Crop Management\crop_model.pkl"
joblib.dump(pipeline, model_path)
print(f"\n✅ Model saved to: {model_path}")
