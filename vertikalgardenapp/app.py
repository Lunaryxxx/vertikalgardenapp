from flask import Flask, request, Response
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('ModelRFR.pkl')

# Define feature names used during training
feature_names = ['TDS_ppm', 'TEMPERATURE_c', 'HUMIDITY', 'pH']

@app.route('/predict', methods=['POST'])
def predict():
    # Get sensor readings from POST request
    data = request.get_json(force=True)
    sensor_readings = data['sensor_readings']

    # Convert list to NumPy array and reshape to 2D
    sensor_readings = np.array(sensor_readings).reshape(1, -1)

    # Create a DataFrame with the sensor readings and feature names
    sensor_readings_df = pd.DataFrame(sensor_readings, columns=feature_names)

    # Make prediction using the Random Forest model
    prediction = model.predict(sensor_readings_df)

    # Round the prediction to the nearest integer
    prediction_rounded = round(prediction[0])

    # Send back the result as plain text with the desired format
    return Response(f"{prediction_rounded} Days", mimetype='text/plain')

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
