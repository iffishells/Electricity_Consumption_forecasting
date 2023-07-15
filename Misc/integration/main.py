from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

def load_model_and_scaler(model_path=None, scaler_path=None):
    # Use default paths if not provided
    model_path = model_path or 'model_files/model.pkl'
    scaler_path = scaler_path or 'model_files/scaler.pkl'

    # Load the model from the .pkl file
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    # Load the scaler from the .pkl file
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    return loaded_model, scaler


def predict_next_points(model, scaler, input_data, num_predictions=24):
    # Reshape the input data
    input_data = np.array(input_data).reshape(1, 1, -1)

    # Predict the next points iteratively
    predictions = []
    for _ in range(num_predictions):
        # Predict the next point
        next_point = model.predict(input_data)[:, -1, 0]
        predictions.append(next_point)

        # Append the predicted point to the input data
        input_data = np.append(input_data[:, :, 1:], next_point.reshape(1, 1, 1), axis=2)

    # Rescale the predicted values
    rescaled_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return rescaled_predictions


@app.route('/predict', methods=['POST'])
def predict():
    MODEL_PATH = 'model_files/model.pkl'  # Set the default model path
    SCALER_PATH = 'model_files/scaler.pkl'  # Set the default scaler path

    data = request.json
    input_data = data.get('input_data')

    if len(input_data) != 20:
        return jsonify({'message': 'Input data length should be equal to 20'})

    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    predictions = predict_next_points(model, scaler, input_data)

    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
