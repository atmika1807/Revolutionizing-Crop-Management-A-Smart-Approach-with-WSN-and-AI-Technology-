import pandas as pd

# Load original dataset with error handling
file_path = r"C:\Users\atpt1\New folder\Revolutionizing Crop Management\crop dataset.csv"
df = pd.read_csv(file_path, on_bad_lines='skip')  # Clean load

# Drop rows with any missing data
df.dropna(inplace=True)

# Check class balance
print("🔍 Before fixing:\n", df['action'].value_counts())

# If 'Fertilize' class is missing, inject synthetic samples
if 'Fertilize' not in df['action'].unique():
    print("⚠️ 'Fertilize' not found. Injecting synthetic examples...")

    fertilize_samples = pd.DataFrame([
        {"crop_type": "Wheat", "soil_type": "Clay", "soil_moisture": 45, "temperature": 28, "humidity": 60, "nutrient_level": 1, "soil_pH": 6.2, "action": "Fertilize"},
        {"crop_type": "Rice", "soil_type": "Loamy", "soil_moisture": 50, "temperature": 30, "humidity": 70, "nutrient_level": 1, "soil_pH": 6.5, "action": "Fertilize"},
        {"crop_type": "Corn", "soil_type": "Sandy", "soil_moisture": 55, "temperature": 32, "humidity": 65, "nutrient_level": 1, "soil_pH": 6.8, "action": "Fertilize"},
        {"crop_type": "Soybean", "soil_type": "Peaty", "soil_moisture": 40, "temperature": 27, "humidity": 58, "nutrient_level": 1, "soil_pH": 6.4, "action": "Fertilize"},
        {"crop_type": "Sugarcane", "soil_type": "Loamy", "soil_moisture": 52, "temperature": 31, "humidity": 62, "nutrient_level": 1, "soil_pH": 6.6, "action": "Fertilize"},
    ])

    df = pd.concat([df, fertilize_samples], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Output summary
print("✅ After fixing:\n", df['action'].value_counts())

# Save cleaned dataset
output_path = r"C:\Users\atpt1\New folder\Revolutionizing Crop Management\crop_dataset_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"📁 Cleaned dataset saved to:\n{output_path}")
