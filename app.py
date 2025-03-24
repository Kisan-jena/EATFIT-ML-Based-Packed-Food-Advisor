import os
import io
import cv2
import pytesseract
import re
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
import time
from Diet_plan import recommend_meal, calculate_bmi

app = Flask(__name__)
app.secret_key = os.urandom(24)

# MySQL configurations
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Kisanjena@123'
app.config['MYSQL_DB'] = 'user_database'

mysql = MySQL(app)
bcrypt = Bcrypt(app)

# Configuration for file upload
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load and preprocess the dataset
# try:
#     data = pd.read_csv("foodpr_cleaned_dataset (3).csv")
# except FileNotFoundError:
#     print("Dataset file not found. Please check the file path.")

# Strip extra spaces from column names
# data.columns = data.columns.str.strip()

# # Preprocess the data
# features = ['fat_100g', 'carbohydrates', 'protein', 'energy_kcal_100g', 'nutriscore_score', 'food_groups']

# # Check if the features are available in the dataset
# missing_columns = [col for col in features if col not in data.columns]
# if missing_columns:
#     print(f"Missing columns in the dataset: {missing_columns}")
# else:
#     X = data[features]
#     y = data['product_name']  # Target: Product name (or any unique product identifier)

#     # Train RandomForest
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf.fit(X_train, y_train)

# Function to check allowed file types
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Nutri-score calculation
# Image enhancement for better OCR
def enhance_image(image):
    height, width = image.shape[:2]
    if width < 640 or height < 480:
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(12,12))
    l = clahe.apply(l)
    
    gamma = 1.2
    l = np.power(l / 255.0, gamma) * 255.0
    l = l.astype('uint8')

    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    denoised = cv2.fastNlMeansDenoisingColored(
        enhanced, 
        None,
        h=7,
        hColor=7,
        templateWindowSize=9,
        searchWindowSize=25
    )
    
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    debug_dir = './debug_images'
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    cv2.imwrite(f"{debug_dir}/enhanced_{int(time.time())}.jpg", thresh)
    
    return thresh

OCR_CONFIGS = [
    # OCR Engine Mode (OEM) and Page Segmentation Mode (PSM) configurations
    # Config 1: Optimized for columnar data (e.g., nutrition tables)
    # OEM 1 = Legacy engine only | PSM 4 = Assume single column of variable-sized text
    {
        'oem': 1,  # Traditional OCR engine (faster for structured layouts)
        'psm': 4,   # Column-based text analysis (ideal for nutrition labels)
    },
    
    # Config 2: General paragraph processing 
    # OEM 3 = LSTM neural net only | PSM 6 = Assume uniform text block
    {
        'oem': 3,  # Advanced LSTM engine (better for continuous text)
        'psm': 6,   # Single text block processing (good for ingredient lists)
    },
    
    # Config 3: Single-line text focus
    # OEM 3 = LSTM engine | PSM 7 = Treat as single text line
    {
        'oem': 3,  # Neural network for character recognition
        'psm': 7,   # Optimized for product names/header text
    },
    
    # Config 4: Sparse text detection
    # OEM 3 = LSTM engine | PSM 11 = Sparse text with orientation detection
    {
        'oem': 3,  # Best accuracy for scattered text
        'psm': 11,  # Finds text fragments (e.g., warning labels, icons)
    },
    
    # Config 5: Multi-column legacy processing
    # OEM 1 = Legacy engine | PSM 3 = Fully automatic layout analysis
    {
        'oem': 1,  # Fallback to traditional OCR
        'psm': 3,   # Auto-detect complex layouts (backup for mixed formats)
    }
]

# OCR and nutrition parsing functions
def enhanced_ocr(processed_image):
    all_text = []
    for cfg in OCR_CONFIGS:
        try:
            text = pytesseract.image_to_string(
                processed_image,
                config=f'--oem {cfg["oem"]} --psm {cfg["psm"]} -c preserve_interword_spaces=1',
                lang='eng'
            )
            all_text.append(text.strip())
        except Exception as e:
            print(f"OCR Error (OEM {cfg['oem']}, PSM {cfg['psm']}): {str(e)}")
    
    return '\n'.join(sorted(set(all_text), key=len, reverse=True))

def extract_text(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        processed = enhance_image(img)
        combined_text = enhanced_ocr(processed)
        
        clean_text = re.sub(r'[^a-zA-Z0-9\s.,%]', '', combined_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text.lower()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return ""

def parse_nutrition(text):
    # Energy patterns with parentheses support
    energy_matches = re.findall(
        r'(?:Energy\s*\(?kcal\)?.*?)(\d+\.?\d*)|'
        r'(\d+\.?\d*)\s*\(?kcal\)?(?=\s|$)',
        text,
        re.IGNORECASE
    )
    
    energy_values = [float(m[0] or m[1]) for m in energy_matches if any(m)]
    
    nutrition = {
        'energy_kcal': energy_values[0] if energy_values else None,
        'sugars': None,
        'salt': None
    }

    sugar_patterns = [
        r'(of\s*which\s*sugars.*?)(\d+\.?\d*)\s*g',
        r'\bsugars?\b.*?(\d+\.?\d*)\s*g'
    ]
    for pattern in sugar_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                nutrition['sugars'] = float(match.group(1))
                break
            except:
                continue

    sodium_match = re.search(r'sodium.*?(\d+\.?\d*)\s*mg', text, re.IGNORECASE)
    if sodium_match:
        try:
            sodium_mg = float(sodium_match.group(1))
            nutrition['salt'] = sodium_mg / 400  # Convert to grams
        except:
            pass
    
    salt_match = re.search(r'salt.*?(\d+\.?\d*)\s*g', text, re.IGNORECASE)
    if salt_match and not nutrition['salt']:
        try:
            nutrition['salt'] = float(salt_match.group(1))
        except:
            pass

    nutrient_patterns = {
        'fat': r'(Total Fat|Fat)[^\d]*(\d+\.?\d*)',
        'saturated_fat': r'(Saturates|Saturated Fat)[^\d]*(\d+\.?\d*)',
        'carbohydrates': r'(Carbohydrates|Carbs)[^\d]*(\d+\.?\d*)',
        'fiber': r'(Fibre|Fiber)[^\d]*(\d+\.?\d*)',
        'protein': r'Protein[^\d]*(\d+\.?\d*)'
    }
    
    for nutrient, pattern in nutrient_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                nutrition[nutrient] = float(match.group(2))
            except:
                continue
    
    return nutrition

def calculate_nutri_score(nutrition):
    """Calculate nutrition score for Indian context (A=best, E=worst)"""
    
    # Negative components (limit 40 points)
    negative_points = 0
    # Energy: FSSAI uses 2000kcal daily reference
    negative_points += min(10, nutrition.get('energy_kcal', 0) / 200)  # Per 200kcal
    # Saturated fat: Align with HFSS thresholds
    negative_points += min(10, nutrition.get('saturated_fat', 0) / 2)  # Per 2g
    # Sugars: Adjusted for Indian sweets/diabetes risk
    negative_points += min(10, nutrition.get('sugars', 0) / 5)         # Per 5g
    # Salt: WHO India recommendation (lower than global)
    negative_points += min(10, nutrition.get('salt', 0) / 0.05)        # Per 0.05g
    
    # Positive components (limit 15 points)
    positive_points = 0
    # Fiber: Account for traditional high-fiber diets
    positive_points += min(7, nutrition.get('fiber', 0) / 1.5)         # Per 1.5g
    # Protein: Higher weight for vegetarian sources
    positive_points += min(8, nutrition.get('protein', 0) / 2)         # Per 2g
    

    # Final calculation (adjusted scale)
    nutri_score = int(negative_points - positive_points)

    # Indian grading scale
    if nutri_score <= -5:
        return 5  # A - Excellent
    elif -4 <= nutri_score <= 3:
        return 4  # B - Good  
    elif 4 <= nutri_score <= 10:
        return 3  # C - Moderate
    elif 11 <= nutri_score <= 18:
        return 2  # D - Unhealthy
    else:
        return 1  # E - Avoid

def process_with_config(image_path, config_idx):
    img = cv2.imread(image_path)
    processed = enhance_image(img)
    
    cfg = OCR_CONFIGS[config_idx]
    text = pytesseract.image_to_string(
        processed,
        config=f'--oem {cfg["oem"]} --psm {cfg["psm"]}',
        lang='eng'
    )
    
    return parse_nutrition(text.lower())

# Routes for user management
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        if user:
            flash('Password reset instructions have been sent to your email.', 'info')
        else:
            flash('Email address not found.', 'danger')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')

@app.route('/get_started')
def get_started():
    if 'user_id' in session:
        return redirect(url_for('landing_page'))
    return redirect(url_for('login'))

@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            flash('Email already exists. Please use a different email address.', 'danger')
            return redirect(url_for('signup'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", 
                    (username, email, hashed_password))
        mysql.connection.commit()
        cur.close()
        flash('You have successfully signed up!', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cur = mysql.connection.cursor()
        
        # First check if user exists at all
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        
        if not user:
            # User doesn't exist in users table
            flash('Please sign up first', 'danger')
            cur.close()
            return redirect(url_for('login'))
        
        # Now verify password for existing user
        if not bcrypt.check_password_hash(user[3], password):
            flash('Invalid password', 'danger')
            cur.close()
            return redirect(url_for('login'))
        
        # User exists and password is correct - check health data
        session['user_id'] = user[0]
        session['user_email'] = user[2]
        
        # Check for existing health data
        cur.execute("SELECT * FROM health_data WHERE user_id = %s", (user[0],))
        health_data = cur.fetchone()
        cur.close()
        
        flash('Login successful!', 'success')
        
        if health_data:
            return redirect(url_for('landing_page'))
        else:
            return redirect(url_for('health_form'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing_page'))

@app.route('/health_form')
def health_form():
    if 'user_id' not in session:
        flash('Please log in to access the health form.', 'warning')
        return redirect(url_for('login'))
    return render_template('healthForm.html')
@app.route('/edit_health_data')
def edit_health_data():
    if 'user_id' not in session:
        flash('Please log in to edit health data', 'warning')
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM health_data WHERE user_id = %s", (user_id,))
    health_data = cur.fetchone()
    cur.close()
    
    if not health_data:
        flash('No health data found. Please create new entry.', 'info')
        return redirect(url_for('health_form'))
    
    return render_template('healthForm.html', health_data=health_data)

@app.route('/submit_health_data', methods=['POST'])
def submit_health_data():
    if 'user_id' not in session:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    cur = None  # Initialize cursor for finally block
    
    try:
        # Validate and parse form data
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        age = int(request.form['age'])
        diabetes = request.form['diabetes']
        bp = request.form['bp']
        cholesterol = request.form['cholesterol']
        
        # Calculate BMI
        height_m = height * 0.3048
        bmi = round(weight / (height_m ** 2), 2)
        
        cur = mysql.connection.cursor()
        
        # Check for existing health data
        cur.execute("SELECT id FROM health_data WHERE user_id = %s", (user_id,))
        existing_data = cur.fetchone()
        
        if existing_data:
            # Update existing record
            cur.execute("""
                UPDATE health_data 
                SET height = %s,
                    weight = %s,
                    bmi = %s,
                    age = %s,
                    diabetes = %s,
                    bp = %s,
                    cholesterol = %s
                WHERE user_id = %s
            """, (height, weight, bmi, age, diabetes, bp, cholesterol, user_id))
            flash_message = 'Health data updated successfully!'
        else:
            # Create new entry
            cur.execute("""
                INSERT INTO health_data 
                (user_id, height, weight, bmi, age, diabetes, bp, cholesterol)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (user_id, height, weight, bmi, age, diabetes, bp, cholesterol))
            flash_message = 'Health data submitted successfully!'
        
        mysql.connection.commit()
        flash(flash_message, 'success')
        return redirect(url_for('profile'))
    
    except ValueError:
        flash('Invalid input for height, weight, or age.', 'danger')
        return redirect(url_for('health_form'))
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Error processing health data: {str(e)}', 'danger')
        return redirect(url_for('health_form'))
    finally:
        if cur:
            cur.close()

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('Access denied, please log in.', 'warning')
        return redirect(url_for('landing_page'))
    user_id = session['user_id']
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET username = %s, email = %s WHERE id = %s", (name, email, user_id))
        if 'profile_image' in request.files:
            image = request.files['profile_image']
            if image and allowed_file(image.filename):
                image_data = image.read()
                cur.execute("UPDATE users SET profile_image = %s WHERE id = %s", (image_data, user_id))
        mysql.connection.commit()
        cur.close()
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('profile'))
    
    cur = mysql.connection.cursor()
    cur.execute("SELECT username, email, profile_image FROM users WHERE id = %s", [user_id])
    user_data = cur.fetchone()
    cur.execute("SELECT * FROM health_data WHERE user_id = %s ORDER BY id DESC LIMIT 1", (user_id,))
    health_info = cur.fetchone()
    cur.close()
    
    username, email, profile_image = user_data
    image_url = url_for('get_profile_image') if profile_image else url_for('static', filename='default-avatar.png')
    
    return render_template('profile.html', username=username, email=email, image_url=image_url, health_info=health_info)

@app.route('/get_profile_image')
def get_profile_image():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    cur = mysql.connection.cursor()
    cur.execute("SELECT profile_image FROM users WHERE id = %s", [user_id])
    profile_image = cur.fetchone()[0]
    cur.close()
    if profile_image:
        return send_file(io.BytesIO(profile_image), mimetype='image/png')
    else:
        return redirect(url_for('static', filename='default-avatar.png'))

# Routes for file upload and nutrition analysis
@app.route("/upload_file", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(upload_path)
            session['file_path'] = upload_path
            session['filename'] = filename
            session['current_config_idx'] = 0
            session['nutrition'] = process_with_config(upload_path, 0)
            return redirect(url_for('verify_extraction'))
            
        except Exception as e:
            return render_template("upload.html", error=str(e))

    return render_template("upload.html")

@app.route("/verify", methods=["GET", "POST"])
def verify_extraction():
    if 'file_path' not in session or 'filename' not in session:
        return redirect(url_for('upload_file'))
    
    if request.method == 'POST':
        if request.form.get('user_response') == 'accept':
            nutrition = session.get('nutrition', {})
            fields = [
                'energy_kcal', 'fat', 'saturated_fat',
                'carbohydrates', 'sugars', 'fiber',
                'protein', 'salt'
            ]
            
            for field in fields:
                value = request.form.get(field)
                if value:
                    try:
                        nutrition[field] = float(value)
                    except ValueError:
                        flash(f'Invalid value for {field.replace("_", " ")}')
                        return redirect(url_for('verify_extraction'))
            
            session['nutrition'] = nutrition
            session['nutri_grade'] = calculate_nutri_score(nutrition)
            return redirect(url_for('product_details'))
        else:
            current_idx = session.get('current_config_idx', 0)
            new_idx = (current_idx + 1) % len(OCR_CONFIGS)
            
            try:
                session['current_config_idx'] = new_idx
                session['nutrition'] = process_with_config(
                    session['file_path'],
                    new_idx
                )
            except Exception as e:
                flash(f"Error processing image: {str(e)}")
                return redirect(url_for('verify_extraction'))
    
    return render_template(
        "verify.html",
        image=session['filename'],
        nutrition=session.get('nutrition', {}),
        config_number=session.get('current_config_idx', 0) + 1
    )

@app.route("/product_details")
def product_details():
    if 'filename' not in session or 'nutrition' not in session:
        return redirect(url_for('upload_file'))
    
    return render_template("product_details.html", 
                      image=session['filename'],
                      nutrition=session.get('nutrition', {}),
                      score=session.get('nutri_grade'))

@app.route("/diet_recommendation")
def diet_recommendation():
    return redirect(url_for('diet_plan'))

@app.route("/cart")
def cart():
    pass

@app.route('/alternative_products')
def alternative_products():
    alternatives = [
        {
            'product_name': 'Product 1',
            'image_url': 'https://example.com/product1.jpg',
            'nutriscore_grade': 'A',
            'reason': 'Based on Nutri-Score similarity'
        },
        {
            'product_name': 'Product 2',
            'image_url': 'https://example.com/product2.jpg',
            'nutriscore_grade': 'B',
            'reason': 'Based on Nutri-Score similarity'
        }
    ]
    return render_template("alternative_products.html", alternatives=alternatives)
###MEAL_RECOMMENDATION
@app.route("/", methods=["GET"])
def meal_page():
    return render_template("meal.html")

@app.route("/recommend_meal", methods=["POST"])
def get_meal():
    
    try:
        data = request.get_json()
        age = int(data["age"])
        weight = float(data["weight"])
        height = float(data["height"])
        disease = data["disease"].strip()

        print(f"🔍 User Input -> Age: {age}, Weight: {weight}, Height: {height}, Disease: '{disease}'")

        bmi, bmi_category, breakfast, lunch, dinner = recommend_meal(age, weight, height, disease)

        return jsonify({
            "bmi": round(bmi, 2),
            "bmi_category": bmi_category,
            "breakfast": breakfast,
            "lunch": lunch,
            "dinner": dinner
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)})

# Routes for diet plan
@app.route('/diet_plan')
def diet_plan():
    if 'user_id' not in session:
        flash('Please login first')
        return redirect(url_for('login'))
    
    # Get user's health data from database
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT height, weight, age, bmi 
        FROM health_data 
        WHERE user_id = %s
    """, (session['user_id'],))
    health_data = cur.fetchone()
    cur.close()
    
    if not health_data:
        flash('Please complete your health profile first')
        return redirect(url_for('health_form'))
    
    return render_template('diet_plan.html', 
                         height=health_data[0],
                         weight=health_data[1],
                         age=health_data[2],
                         bmi=health_data[3])

@app.route('/get_diet_plan', methods=['POST'])
def get_diet_plan():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    try:
        disease = request.form.get('disease', 'none')
        height = float(request.form.get('height'))  # Height in feet
        weight = float(request.form.get('weight'))
        age = int(request.form.get('age'))
        
        # Get meal recommendations
        recommendations = recommend_meal(age, weight, height, disease)
        
        return jsonify({
            'breakfast': recommendations['breakfast'],
            'lunch': recommendations['lunch'],
            'dinner': recommendations['dinner']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if not all(os.path.exists(f) for f in ["rf_breakfast.pkl", "rf_lunch.pkl", "rf_dinner.pkl", "label_encoders.pkl"]):
        print("❌ Model files missing! Ensure .pkl files are in the project folder.")
    else:
        print("✅ All model files are present.")
    app.run(debug=True, port=5001)
