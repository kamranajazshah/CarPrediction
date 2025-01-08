from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd  # Import pandas

# Initialize Flask app
app = Flask(__name__)

# Load the trained model pipeline
MODEL_PATH = 'car prediction project/LinearRegression.pkl'
with open(MODEL_PATH, 'rb') as model_file:
    model_pipeline = pickle.load(model_file)

# Define the expected column names for the model
COLUMN_NAMES = ['name', 'company', 'year', 'kms_driven', 'fuel_type']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form['name']
            company = request.form['company']
            year = int(request.form['year'])
            kms_driven = float(request.form['kms_driven'])
            fuel_type = request.form['fuel_type']

            # Prepare input features as a pandas DataFrame
            input_features = pd.DataFrame([[name, company, year, kms_driven, fuel_type]], columns=COLUMN_NAMES)

            # Make prediction
            prediction = model_pipeline.predict(input_features)

            # Render template with prediction
            return render_template('index.html', prediction=f"Predicted Price: {prediction[0]:.2f}")
        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request
        data = request.get_json()
        
        # Extract input features and create a pandas DataFrame
        input_features = pd.DataFrame([[
            data['name'],
            data['company'],
            data['year'],
            data['kms_driven'],
            data['fuel_type']
        ]], columns=COLUMN_NAMES)

        # Make prediction
        prediction = model_pipeline.predict(input_features)

        # Return prediction as JSON
        response = {'predicted_price': prediction[0]}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
