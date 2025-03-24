import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("EATFIT_DIET.csv")

# Convert Height to Meters & Calculate BMI
df["Height (m)"] = df["Height (ft)"] * 0.3048  
df["BMI"] = df["Weight (kg)"] / (df["Height (m)"] ** 2)

# Categorize BMI
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

df["BMI_Category"] = df["BMI"].apply(categorize_bmi)

# Encode Categorical Columns
label_encoders = {}
categorical_cols = ["Diseases", "BMI_Category", "Breakfast", "Lunch", "Dinner"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # Convert to string before encoding
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & Targets
X = df[["Age", "Weight (kg)", "Height (m)", "BMI", "Diseases"]]
y_breakfast = df["Breakfast"]
y_lunch = df["Lunch"]
y_dinner = df["Dinner"]

# Split Data
X_train, X_test, yb_train, yb_test = train_test_split(X, y_breakfast, test_size=0.2, random_state=42)
_, _, yl_train, yl_test = train_test_split(X, y_lunch, test_size=0.2, random_state=42)
_, _, yd_train, yd_test = train_test_split(X, y_dinner, test_size=0.2, random_state=42)

# Train Random Forest Models
rf_breakfast = RandomForestClassifier(n_estimators=200, random_state=42)
rf_breakfast.fit(X_train, yb_train)

rf_lunch = RandomForestClassifier(n_estimators=200, random_state=42)
rf_lunch.fit(X_train, yl_train)

rf_dinner = RandomForestClassifier(n_estimators=200, random_state=42)
rf_dinner.fit(X_train, yd_train)

# Save Models & Encoders
joblib.dump(rf_breakfast, "rf_breakfast.pkl")
joblib.dump(rf_lunch, "rf_lunch.pkl")
joblib.dump(rf_dinner, "rf_dinner.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Models trained and saved successfully!")
