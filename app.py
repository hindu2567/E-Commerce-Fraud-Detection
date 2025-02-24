from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import pickle
import subprocess
import logging
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For session management

# Load the trained Stacking Classifier model if available
try:
    stacking_model = pickle.load(open('model/stacking_model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    stacking_model = None

@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Static login check
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Clear session data to log out the user
    session.pop('logged_in', None)
    return redirect(url_for('login'))  # Redirect to login page after logout

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    if request.method == 'POST':
        file = request.files['dataset']
        if file:
            file.save('data/e_commerce_data.csv')
            # You can preview the dataset here
            df = pd.read_csv('data/e_commerce_data.csv')
            table_html = df.to_html(classes='data', index=False)
            return render_template('upload.html', filename=file.filename, tables=table_html)

    return render_template('upload.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # Execute the model training script
        result = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Model training successful")
        else:
            print(f"Error in training: {result.stderr}")
    except Exception as e:
        print(f"Error during model training: {e}")
        return render_template('upload.html', error="Model training failed.")
    
    return redirect(url_for('upload'))  # Redirect back to the upload page after training

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    if request.method == 'POST':
        try:
            # Extract and validate input data, including transaction_id and customer_id
            required_fields = ['transaction_id', 'customer_id', 'transaction_amount', 'payment_method', 'product_category', 
                               'quantity', 'customer_age', 'device_used', 'account_age_days', 'transaction_hour']
            input_data = []
            for field in required_fields:
                value = request.form.get(field)
                if not value:
                    return render_template('predict.html', error=f"Missing value for {field}.")
                input_data.append(float(value) if field not in ['transaction_id', 'customer_id'] else int(value))

            input_array = np.array([input_data])

            if stacking_model:
                prediction = stacking_model.predict(input_array)[0]
                result = "Fraud" if prediction == 1 else "Not Fraud"
                return render_template('predict.html', prediction=result,
                                       transaction_id=request.form['transaction_id'],
                                       customer_id=request.form['customer_id'],
                                       transaction_amount=request.form['transaction_amount'],
                                       payment_method=request.form['payment_method'],
                                       product_category=request.form['product_category'],
                                       quantity=request.form['quantity'],
                                       customer_age=request.form['customer_age'],
                                       device_used=request.form['device_used'],
                                       account_age_days=request.form['account_age_days'],
                                       transaction_hour=request.form['transaction_hour'])
            else:
                return render_template('predict.html', error="Model not loaded.")
        
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return render_template('predict.html', error=f"Error: {e}")

    return render_template('predict.html')

@app.route('/performance_analysis')
def performance_analysis():
    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in
    return render_template('analysis.html')

if __name__ == 'main':
    app.run(debug=True)