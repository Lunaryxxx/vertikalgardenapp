from flask import Flask, request, Response
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained Random Forest model and scaler
model = joblib.load('tuned_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get sensor readings from POST request
    data = request.get_json(force=True)
    sensor_readings = data['sensor_readings']

    # Convert list to NumPy array and reshape to 2D
    sensor_readings = np.array(sensor_readings).reshape(1, -1)

    # Standardize the sensor readings using the same scaler used for training
    sensor_readings_scaled = scaler.transform(sensor_readings)

    # Make prediction using the Random Forest model
    prediction = model.predict(sensor_readings_scaled)

    # Send back the result as plain text with the desired format
    return Response(f"Predicted Condition: {prediction[0]}", mimetype='text/plain')

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
