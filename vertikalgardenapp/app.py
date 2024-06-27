from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from google.cloud import firestore

app = Flask(_name_)

# Initialize Firestore client
db = firestore.Client()

# Load your trained model
model = tf.keras.models.load_model('harvest_prediction_model.h5')

def get_sensor_readings_from_firestore(document_id):
    # Fetch the document from Firestore
    doc_ref = db.collection('sensor_readings').document(document_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get('sensor_readings', [])
    else:
        return []

@app.route('/predict', methods=['POST'])
def predict():
    # Get Firestore document ID from POST request
    data = request.get_json(force=True)
    document_id = data['document_id']

    # Fetch sensor readings from Firestore
    sensor_readings = get_sensor_readings_from_firestore(document_id)

    if not sensor_readings:
        return jsonify(error="No sensor readings found"), 400

    # Convert list to NumPy array and reshape to 2D
    sensor_readings = np.array(sensor_readings).reshape(1, -1)

    # Make prediction using your model
    prediction = model.predict(sensor_readings)

    # Round the prediction to the nearest integer
    prediction_rounded = round(prediction[0][0])

    # Send back the result as JSON
    return jsonify(prediction=int(prediction_rounded))

if _name_ == '_main_':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
