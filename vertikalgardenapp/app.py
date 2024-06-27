from flask import Flask, request, Response
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

app = Flask(__name__)

# Define your model architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
model = SimpleNN()

# Load the saved model weights
model.load_state_dict(torch.load('harvest_prediction_model.pth'))
model.eval()

# Load the scaler
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get sensor readings from POST request
    data = request.get_json(force=True)
    sensor_readings = data['sensor_readings']

    # Convert list to NumPy array and reshape to 2D
    sensor_readings = np.array(sensor_readings).reshape(1, -1)
    
    # Scale the sensor readings
    sensor_readings = scaler.transform(sensor_readings)

    # Convert the readings to a PyTorch tensor
    sensor_readings = torch.tensor(sensor_readings, dtype=torch.float32)

    # Make prediction using the model
    with torch.no_grad():
        prediction = model(sensor_readings)

    # Round the prediction to the nearest integer
    prediction_rounded = round(prediction.item())

    # Send back the result as plain text with the desired format
    return Response(f"{prediction_rounded} Days", mimetype='text/plain')

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
