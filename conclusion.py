import pandas as pd
import os
import sys
import re
import mysql.connector
import requests
from flask import Flask, request, render_template



sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(r"C:/Users/Priyanka/Downloads/EATFIT-review-1/EATFIT-review-1/nutrients-dataset.csv", encoding="utf-8-sig")
def db_connect():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="root",
        database="user_database"
    )
    cursor = conn.cursor(dictionary=True) 
    return conn, cursor
def fetch_health_data(username):
    conn = db_connect()
    cursor = conn.cursor(dictionary=True)
    
    query = """
    SELECT h.height, h.weight, h.bmi, h.age, h.diabetes, h.bp, h.cholesterol
    FROM health_data h
    JOIN users u ON h.user_id = u.id
    WHERE u.username = %s
    """
    cursor.execute(query, (username,))
    health_data = cursor.fetchone()
    
    cursor.close()
    conn.close()
    return health_data
def fetch_nutrients(barcode):
    api_url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    try:
        response = requests.get(api_url)
        data = response.json()
        
        if data.get("status") == 1:  # Product found
            nutriments = data["product"].get("nutriments", {})
            return {
                "fat": nutriments.get("fat_100g", 0),
                "saturated_fat": nutriments.get("saturated-fat_100g", 0),
                "carbohydrates": nutriments.get("carbohydrates_100g", 0),
                "sugar": nutriments.get("sugars_100g", 0),
                "fiber": nutriments.get("fiber_100g", 0),
                "protein": nutriments.get("proteins_100g", 0),
                "salt": nutriments.get("salt_100g", 0),
                "cholesterol": nutriments.get("cholesterol_100g", 0),
                "trans_fat": nutriments.get("trans-fat_100g", 0)
            }
        else:
            return None
    except Exception as e:
        return None
def get_age_column(username):
    health_data = fetch_health_data(username)
    if not health_data:
        return None  # Return None if user not found
    
    age = health_data["age"]

    if 0 <= age <= 6:
        return "0-6 years"
    elif 7 <= age <= 12:
        return "7-12 years"
    elif 13 <= age <= 18:
        return "13-18 years"
    else:
        return "Adults"

def categorize_bmi(username):
    health_data = fetch_health_data(username)
    if not health_data:
        return None  # Return None if user not found
    
    bmi = health_data["bmi"]    
    if bmi < 18.5:
        return "Underweight (BMI < 18.5)"
    elif 18.5 <= bmi <= 24.9:
        return "Normal (BMI 18.5-24.9)"
    elif 25 <= bmi <= 29.9:
        return "Overweight (BMI 25-29.9)"
    else:
        return "Obese (BMI 30+)"

def extract_numeric(value):
    return float(re.sub(r"[^\d.]", "", value))

def check_product_safety(nutrition, health_data):
    print("Received Nutrition Data:", nutrition)
    print("Received Health Data:", health_data)

    age_column = get_age_column(health_data["age"])
    print("Age Column Selected:", age_column)

    review = []

    for nutrient, value in nutrition.items():
        print(f"Checking: {nutrient} with value {value}")

        row = df[df["Nutrient/chemicals to avoid"].str.lower() == nutrient.replace("_", " ")]
        if not row.empty:
            limit_str = str(row[age_column].values[0]).strip()
            print(f"Limit for {nutrient}: {limit_str}")

            # Extract numeric values correctly
            if "avoid" in limit_str.lower() or limit_str == "0" or re.match(r"0\s*[gmg]*", limit_str, re.IGNORECASE):
                limit = 0  # Explicitly set limit to 0 for "Avoid" cases
            else:
                if "-" in limit_str:  # Handling range values like "≤ 10-15g"
                    parts = limit_str.split("-")
                    lower_bound = extract_numeric(parts[0])
                    upper_bound = extract_numeric(parts[1]) if len(parts) > 1 else lower_bound
                    limit = upper_bound  # Take the higher end of the range for ≤ comparisons
                else:
                    limit = extract_numeric(limit_str)

            # Now apply the conditions correctly
            if limit_str.startswith("≤") or limit == 0:
                if value > limit:
                    review.append(f"{nutrient} exceeds limit ({value}g > {limit}g), hence this product is not recommended for you.")
            elif limit_str.startswith("≥"):
                if value < limit:
                    review.append(f"{nutrient} is below recommended ({value}g < {limit}g).")

    if not review:
        return "All nutrients are within safe limits."

    return " ".join(review)

