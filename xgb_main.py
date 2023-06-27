from flask import Flask, request, jsonify
import pickle
import xgboost as xgb
from darts.models.forecasting.xgboost import XGBModel 

app = Flask(__name__)

# Load the XGBoost model from the saved file
model_loaded = XGBModel.load("xgbModel.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    print(request.json)
    data = request.json
    input_data = data.get('input_data')
    print(input_data)
    # print('predict function has been called')
    # # print(request)
    # # try:
    # # Get the input data from the request
    # input_data = request.get_json()
    # print('Input Data Recived : ',input_data)
    # # Perform prediction using the loaded model
    # predicted = model_loaded.predict(input_data['series'], ntree_limit=342)
    # print(predict)
    # # Return the prediction as a response
    # response = {'predicted': predicted.tolist()}

    # return jsonify(response)

    # except Exception as e:
    #     return jsonify({'error': str(e)}), 400
    return input_data
if __name__ == '__main__':
    app.run(debug=True,port=40002)
