# Harvest Prediction API

This repository hosts a Flask API for predicting harvest outcomes based on sensor readings using a Random Forest model.

## Overview

# Vertikal Garden Prediction API

This repository contains a Flask-based API for predicting the optimality of conditions in a vertical garden based on sensor readings. The model used is a pre-trained Random Forest model saved using `joblib`.

## Requirements

- Python 3.x
- Flask
- joblib
- pandas

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/vertikalgarden-predictor.git
    cd vertikalgarden-predictor
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install Flask joblib pandas
    ```

4. Ensure that the pre-trained model `vertikalgardenrfv1.pkl` is in the project directory.

## Usage

1. Run the Flask app:

    ```bash
    python app.py
    ```

2. The API will be available at `http://127.0.0.1:5000/predict`.

## API Endpoint

### POST /predict

#### Request

The endpoint expects a JSON object containing sensor readings.

- **sensor_readings**: A list of 4 numerical values representing the sensor readings in the following order:
  - TDS
