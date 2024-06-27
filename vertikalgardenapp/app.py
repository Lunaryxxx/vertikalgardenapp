from flask import Flask, request, Response
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('harvest_prediction_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get sensor readings from POST request
    data = request.get_json(force=True)
    sensor_readings = data['sensor_readings']

    # Convert list to NumPy array and reshape to 2D
    sensor_readings = np.array(sensor_readings).reshape(1, -1)

    # Make prediction using your model
    prediction = model.predict(sensor_readings)

    # Round the prediction to the nearest integer
    prediction_rounded = round(prediction[0][0])

    # Send back the result as plain text with the desired format
    return Response(f"Prediction: {prediction_rounded} Days", mimetype='text/plain')

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
