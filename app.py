"""from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Convert data into an array and reshape for prediction
    input_data = np.array([data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                           data['SkinThickness'], data['Insulin'], data['BMI'],
                           data['DiabetesPedigreeFunction'], data['Age']])
    input_data = input_data.reshape(1, -1)
    
    # Scale the data
    input_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_data)
    result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)"""
from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np

# Existing code...


app = Flask(__name__)

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Convert data into an array and reshape for prediction
    input_data = np.array([data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                           data['SkinThickness'], data['Insulin'], data['BMI'],
                           data['DiabetesPedigreeFunction'], data['Age']])
    input_data = input_data.reshape(1, -1)
    
    # Scale the data
    input_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_data)
    result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    
    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)"""




