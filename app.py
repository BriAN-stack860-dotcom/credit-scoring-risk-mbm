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

        # Use threshold = 0.1 for high recall
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
