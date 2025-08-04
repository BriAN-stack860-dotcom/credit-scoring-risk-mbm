<<<<<<< HEAD
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = load_model('credit_risk_model.h5')
scaler = joblib.load('scaler.pkl')

# Feature names (same order as during training)
feature_names = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
    'TotalLatePayments', 'MonthlyIncome_was_missing',
    'NumberOfDependents_was_missing'
]

@app.route('/')
def home():
    return render_template('index.html')  # Optional: HTML form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract features in correct order
        input_data = []
        for feature in feature_names:
            value = data.get(feature, 0)
            input_data.append(float(value))

        # Convert to numpy array and scale
        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)

        # Predict probability
        risk_prob = model.predict(input_scaled)[0][0]

        # Apply threshold (0.1) for high recall
        prediction = 1 if risk_prob > 0.1 else 0

        # Return result
        return jsonify({
            'risk_score': float(risk_prob),
            'prediction': int(prediction),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model and scaler
try:
    model = load_model('credit_risk_model.h5')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model and scaler loaded!")
except Exception as e:
    print("❌ Error loading model/scaler:", e)

# Feature names (must match training)
feature_names = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
    'TotalLatePayments', 'MonthlyIncome_was_missing',
    'NumberOfDependents_was_missing'
]

@app.route('/')
def home():
    return ('index.html')  # Serve the UI

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features in correct order
        input_data = [float(data.get(f, 0)) for f in feature_names]
        input_array = np.array([input_data])

        # Scale and predict
        input_scaled = scaler.transform(input_array)
        risk_prob = model.predict(input_scaled)[0][0]

        # Use threshold = 0.1 for high recall (updated comment)
        prediction = 1 if risk_prob > 0.1 else 0

        return jsonify({
            'risk_score': float(risk_prob),
            'prediction': int(prediction),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


