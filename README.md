# Harvest Prediction API

This repository hosts a Flask API for predicting harvest outcomes based on sensor readings using a Random Forest model.

## Overview

The API loads a pre-trained TensorFlow model (`harvest_prediction_model.h5`) and exposes a single endpoint for making predictions. Sensor readings are sent via a POST request to `/predict`, and the API responds with the predicted harvest outcome rounded to the nearest integer.
