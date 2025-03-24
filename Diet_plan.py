import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load models and encoders
try:
    rf_breakfast = joblib.load("rf_breakfast.pkl")
    rf_lunch = joblib.load("rf_lunch.pkl")
    rf_dinner = joblib.load("rf_dinner.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    print(" Model files loaded successfully!")
    print("Available diseases:", list(label_encoders["Diseases"].classes_))
except FileNotFoundError as e:
    print(f" Error loading models: {e}")
    exit()

def calculate_bmi(weight, height_ft):
    """Calculate BMI and return category."""
    height_m = height_ft * 0.3048
    bmi = weight / (height_m ** 2)

    if bmi < 18.5:
        return bmi, "Underweight"
    elif 18.5 <= bmi < 24.9:
        return bmi, "Normal weight"
    elif 25 <= bmi < 29.9:
        return bmi, "Overweight"
    else:
        return bmi, "Obese"

def recommend_meal(age, weight, height_ft, disease):
    """Predict and return meal recommendations."""
    # Convert height to meters and calculate BMI
    height_m = height_ft * 0.3048
    bmi, bmi_category = calculate_bmi(weight, height_ft)

    # Map common disease names to training data names
    disease_mapping = {
        'obesity': 'obese',
        'cardiovascular': 'heart disease',
        'cardiovascular disease': 'heart disease',
        'heart': 'heart disease',
        'diabetes': 'diabetes',
        'hypertension': 'hypertension',
        'none': 'none'
    }

    # Handle disease encoding
    disease = disease.strip().lower()
    disease = disease_mapping.get(disease, disease)  # Map to standard name or keep original
    
    try:
        disease_encoded = label_encoders["Diseases"].transform([disease])[0]
    except ValueError as e:
        print(f"Warning: Disease '{disease}' not found. Using 'none' instead.")
        disease_encoded = label_encoders["Diseases"].transform(['none'])[0]

    # Prepare input features
    user_data = np.array([[age, weight, height_m, bmi, disease_encoded]])

    # Get predictions
    breakfast_pred = rf_breakfast.predict(user_data)[0]
    lunch_pred = rf_lunch.predict(user_data)[0]
    dinner_pred = rf_dinner.predict(user_data)[0]

    # Decode predictions
    breakfast = label_encoders["Breakfast"].inverse_transform([breakfast_pred])[0]
    lunch = label_encoders["Lunch"].inverse_transform([lunch_pred])[0]
    dinner = label_encoders["Dinner"].inverse_transform([dinner_pred])[0]

    return {
        'breakfast': breakfast,
        'lunch': lunch,
        'dinner': dinner,
        'bmi': round(bmi, 2),
        'bmi_category': bmi_category
    }
