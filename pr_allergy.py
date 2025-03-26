from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd

app = Flask(__name__)

# Load allergy dataset (Ensure the file is in the same directory)
df_allergies = pd.read_csv("food_allergies.csv")  

# Ensure correct column names
df_allergies.rename(columns={"Allergies/Problems Caused": "Allergies"}, inplace=True)

# Function to fetch ingredients from barcode
def fetch_ingredients(barcode):
    api_url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"

    try:
        response = requests.get(api_url)
        data = response.json()

        if data.get("status") == 1:  # Product found
            ingredients_text = data["product"].get("ingredients_text", "")
            ingredient_list = [ing.strip() for ing in ingredients_text.split(",") if ing.strip()]
            return ingredient_list
        else:
            return None

    except Exception as e:
        return None

# Function to map ingredients to allergies
def map_allergens_to_ingredients(ingredients):
    if not ingredients:
        return []

    # Convert all ingredients to lowercase for matching
    matched_allergies = df_allergies[
        df_allergies["Ingredients"].str.lower().isin([ing.lower() for ing in ingredients])
    ]

    if not matched_allergies.empty:
        return matched_allergies[["Ingredients", "Allergies"]].to_dict(orient="records")
    return []

# Route to render HTML page
@app.route("/")
def home():
    return render_template("pr_allergy.html")

# Route to handle barcode check
@app.route("/check_barcode", methods=["POST"])
def check_barcode():
    barcode = request.form.get("barcode")

    if not barcode:
        return jsonify({"error": "Please enter a valid barcode."})

    ingredients = fetch_ingredients(barcode)

    if ingredients is None:
        return jsonify({"error": "Product not found or error fetching data."})

    allergies = map_allergens_to_ingredients(ingredients)

    return jsonify({"ingredients": ingredients, "allergies": allergies})

if __name__ == "__main__":
    app.run(debug=True)
