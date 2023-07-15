import numpy as np

def calculate_errors(predicted, actual):
    # Convert lists to numpy arrays
    predicted = np.array(predicted)
    actual = np.array(actual)

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predicted - actual))

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))

    # Mean Squared Error (MSE)
    mse = np.mean((predicted - actual) ** 2)

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return mae, rmse, mse, mape
# # train_mae , train_rmse , train_mse,train_mape = calculate_errors(reversedPrediction,reversedActual)
# test_mae , test_rmse , test_mse,test_mape = calculate_errors(reversedPrediction,reversedActual)

# print('Results on Training Data')
# print(f'MAE : {train_mae}\nRMSE : {train_rmse}\nMSE : {train_mse}\nMAPE : {train_mape}'.format())