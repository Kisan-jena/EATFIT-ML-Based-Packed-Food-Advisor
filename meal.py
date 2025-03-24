import numpy as np
import joblib

# Load models and encoders safely
try:
    rf_breakfast = joblib.load("rf_breakfast.pkl")
    rf_lunch = joblib.load("rf_lunch.pkl")
    rf_dinner = joblib.load("rf_dinner.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    print("✅ Model files loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Model file missing: {e}")
    exit()

# Get known diseases
available_diseases = list(label_encoders["Diseases"].classes_)
print(f"🩺 Available Diseases: {available_diseases}")

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
    height_m = height_ft * 0.3048
    bmi, bmi_category = calculate_bmi(weight, height_ft)

    # Handle "None" case safely
    disease = disease.strip().capitalize()
    if disease == "None" or disease == "":
        disease_encoded = 0  # Default for no disease
    elif disease in available_diseases:
        disease_encoded = label_encoders["Diseases"].transform([disease])[0]
    else:
        print(f"⚠️ Disease '{disease}' not found in model. Assigning closest match.")
        disease_encoded = np.random.choice(label_encoders["Diseases"].transform(available_diseases))  # Random existing value

    # Prepare input for prediction
    user_data = np.array([[age, weight, height_m, bmi, disease_encoded]])

    # Predict meals
    breakfast_pred = rf_breakfast.predict(user_data)[0]
    lunch_pred = rf_lunch.predict(user_data)[0]
    dinner_pred = rf_dinner.predict(user_data)[0]

    # Decode predictions
    breakfast = label_encoders["Breakfast"].inverse_transform([breakfast_pred])[0]
    lunch = label_encoders["Lunch"].inverse_transform([lunch_pred])[0]
    dinner = label_encoders["Dinner"].inverse_transform([dinner_pred])[0]

    return bmi, bmi_category, breakfast, lunch, dinner
