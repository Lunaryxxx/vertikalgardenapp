from flask import Flask, request, jsonify, Response
import joblib
import pandas as pd
import os

# Inisialisasi Flask app
app = Flask(__name__)

# Tentukan path untuk model yang sudah disimpan
model_path = 'vertikalgardenrfv2.pkl'

# Muat model
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Dapatkan data JSON dari request
        data = request.get_json()

        # Ekstrak sensor readings dari data
        sensor_readings = data.get("sensor_readings")

        # Validasi input data
        if not sensor_readings or len(sensor_readings) != 4:
            return Response('Invalid input. Expecting 4 sensor readings.', status=400, mimetype='text/plain')

        # Ubah sensor readings menjadi DataFrame dengan nama kolom yang sesuai
        input_data = pd.DataFrame([sensor_readings], columns=['TDS_ppm', 'TEMPERATURE_c', 'HUMIDITY', 'pH'])

        # Pastikan urutan kolom sesuai dengan urutan saat melatih model
        input_data = input_data[['TEMPERATURE_c', 'HUMIDITY', 'TDS_ppm', 'pH']]

        # Lakukan prediksi menggunakan model yang telah dimuat
        prediction = model.predict(input_data)

        # Prediksi adalah angka konversinya ke label asli
        # Misalnya: 0 -> "Optimal", 1 -> "Tidak Optimal"
        prediction_label = "Optimal" if prediction[0] == 0 else "Tidak Optimal"

        # Kembalikan hasil prediksi sebagai plain text
        return Response(f"{prediction_label}", mimetype='text/plain')

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500, mimetype='text/plain')

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
