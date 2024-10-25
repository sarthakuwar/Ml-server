from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from joblib import load

# Load the trained model
model = load('demand_model.joblib')

# Initialize Flask app
app = Flask(__name__)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json(force=True)
    
    # Extract features from the JSON request
    past_sales = data.get('past_sales')
    is_festival = data.get('is_festival')
    discount = data.get('discount')
    availability_nearby = data.get('availability_nearby')
    
    # Prepare the input for prediction
    features = np.array([[past_sales, is_festival, discount, availability_nearby]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return prediction as JSON
    return jsonify({'predicted_quantity': prediction[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
