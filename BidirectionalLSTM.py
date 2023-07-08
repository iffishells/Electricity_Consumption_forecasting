# Import necessary libraries
import pandas as pd  # Library for data manipulation and analysis
import numpy as np  # Library for numerical computing
import glob  # Library for file handling
import re  # Library for regular expressions
from tqdm import tqdm  # Library for creating progress bars

import plotly.graph_objects as go  # Library for creating interactive plots

# Preprocessing
from sklearn import preprocessing  # Library for data preprocessing
import keras  # Deep learning library
from keras.models import Sequential  # Sequential model for stacking layers
from keras.layers.core import Dense, Dropout, Activation  # Layers for fully connected neural networks
from keras.layers import LSTM  # LSTM layer for sequence modeling
from keras.models import load_model  # Loading pre-trained models
import matplotlib.pyplot as plt  # Library for basic data visualization
import h5py  # Library for handling large datasets
import datetime  # Library for date and time operations
import tensorflow as tf  # Deep learning library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import os
from joblib import dump, load


# Creating Folder For sasving model files

os.makedirs('ModelFiles',exist_ok=True)

def extract_digits(string):
    # Extract digits from the string using regular expression
    digits = re.findall(r'\d+', string)
    return int(digits[0]) if digits else 0


def perform_downsampling(data, freq, aggregation_func='sum'):
    # Create a copy of the original data to avoid modifying it directly
    df = data.copy()

    # Check if 'Timestamp' column is already a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        # Convert 'Timestamp' column to datetime index
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

    # Downsample the DataFrame to the specified frequency and apply the aggregation function
    downsampled_df = df.resample(freq).agg(aggregation_func)

    # Fill missing values using forward fill
    downsampled_df.fillna(method='ffill', inplace=True)

    # Reset the index of the downsampled DataFrame
    downsampled_df.reset_index(inplace=True)

    return downsampled_df


def plot(df,x_feature_name,y_feature_name,title):
    """
    This function takes two dataframes as input and plots the number of calls per day and per week.

    Args:
    daily_df (pandas.DataFrame): A dataframe containing daily call data.
    weekly_df (pandas.DataFrame): A dataframe containing weekly call data.

    Returns:
    None
    """

    # A new instance of the go.Figure() class from the plotly.graph_objects library is created. This will be used to create the plot
    fig = go.Figure()
    # Add a trace for daily calls
    # A trace is added to the figure using the go.Scatter() class from plotly.graph_objects. 
    # It specifies the x and y data for the plot, assigns a name to the trace, 
    # and sets the mode to display lines and markers.
    fig.add_trace(
        go.Scatter(
            x=df[x_feature_name],
            y=df[y_feature_name],
            name=y_feature_name,
            mode='lines+markers'
        ))

 

    # Update xaxis properties
    # The x-axis and y-axis titles are updated using the update_xaxes() and update_yaxes() methods of the figure object.
    fig.update_xaxes(title_text='Date')

    # Update yaxis properties
    fig.update_yaxes(title_text=y_feature_name)

    # Update title and height
    # The layout of the figure is updated using the update_layout() method. The title, height, and width of the plot are set.
    fig.update_layout(
        title=f'{title}',
        height=500,
        width=1200
    )

    # Show the plot
    # The plot is displayed using the show() method of the figure object.
    fig.show()


def prepare_data(num_input_days, num_forecast_hours,data ):
    # Load the data
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data.set_index('Timestamp')

    # Create the input and output sequences
    input_seq = []
    output_seq = []

    # Normalize the data
    scaler = MinMaxScaler()
    
    # Convert data to numpy array
    values = data['Summe'].values

    values = scaler.fit_transform(values.reshape(-1,1))
      # Save it to a file
    dump(scaler, 'ModelFiles/scaler.pkl')
    
    # Create the input and output sequences
    for i in range(num_input_days, len(values) - num_forecast_hours):
        input_seq.append(values[i - num_input_days:i])
        output_seq.append(values[i:i + num_forecast_hours])

    # Convert the input and output sequences to numpy arrays
    input_seq = np.array(input_seq)
    output_seq = np.array(output_seq)

   
    
  
    # input_seq = scaler.fit_transform(input_seq)
    # output_seq = scaler.fit_transform(output_seq)

    # Split the data into training, validation and testing sets
    input_seq_train_val, input_seq_test, output_seq_train_val, output_seq_test = train_test_split(input_seq, output_seq, test_size=0.2, shuffle=False)
    train_input, val_input, train_output, val_output = train_test_split(input_seq_train_val, output_seq_train_val, test_size=0.25, shuffle=False)

    # Reshape the input sequences to 3D arrays
    train_input = train_input.reshape(-1, num_input_days, 1)
    val_input = val_input.reshape(-1, num_input_days, 1)
    test_input = input_seq_test.reshape(-1, num_input_days, 1)

    return train_input, val_input, test_input, train_output, val_output, output_seq_test

def create_model(num_input_days, lstm_units, dense_units, learning_rate,input_shape):
    # Build the bidirectional LSTM model
    model = Sequential()
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(num_forecast_hours, activation='linear'))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    print(model.summary())
    return model

def train_and_evaluate_model(model, train_input, train_output, val_input, val_output, batch_size, epochs, steps_per_epoch):
    # Train the model
    history = model.fit(train_input, train_output, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, validation_data=(val_input, val_output))

    # Evaluate the model
    loss = model.evaluate(val_input, val_output)
    print('Validation Loss:', loss)

    return history, loss
def plot_loss(history):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def create_features(df, timestamp_col):
    """
    Create time series features from datetime index.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    timestamp_col (str): Name of timestamp column
    
    Returns:
    df (pandas.DataFrame): Dataframe with added features.
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Create new time related features
    df['Year'] = df[timestamp_col].dt.year
    df['Month'] = df[timestamp_col].dt.month
    df['Day'] = df[timestamp_col].dt.day
    df['Hour'] = df[timestamp_col].dt.hour
    df['Minute'] = df[timestamp_col].dt.minute
    df['DayOfWeek'] = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6
    df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # 1 if the day is weekend

    # Consider holidays if your data has specific trends on holidays
    cal = calendar()
    holidays = cal.holidays(start=df[timestamp_col].min(), end=df[timestamp_col].max())
    df['Holiday'] = df[timestamp_col].isin(holidays).astype(int)  # 1 if the day is a US Federal holiday

    # You may also consider to create a feature that represents the time of the day (morning, afternoon, evening, night)
    df['TimeOfDay'] = pd.cut(df['Hour'], bins=[-0.1, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    return df

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def evaluate_model(model, test_input, test_output, scaler):
    # Predict the output for the test input
    test_predictions = model.predict(test_input)

    print(f"test_predictions.shape: {test_predictions.shape}")
    print(f"test_output.shape: {test_output.shape}")

    # Reshape test predictions and test output to be suitable for inverse transform
    test_predictions_2D = test_predictions.reshape((test_predictions.shape[0] * test_predictions.shape[1], 1))
    test_output_2D = test_output.reshape((test_output.shape[0] * test_output.shape[1], 1))
    
    print(f"test_predictions_2D.shape: {test_predictions_2D.shape}")
    print(f"test_output_2D.shape: {test_output_2D.shape}")

    # Inverse transform the predictions
    test_predictions = scaler.inverse_transform(test_predictions_2D)
    test_predictions = test_predictions.reshape(test_output.shape)

    # Inverse transform the actual test output
    test_output_actual = scaler.inverse_transform(test_output_2D)
    test_output_actual = test_output_actual.reshape(test_output.shape)

    print(f"test_predictions.shape after reshaping: {test_predictions.shape}")
    print(f"test_output_actual.shape after reshaping: {test_output_actual.shape}")

    # Calculate Mean Absolute Error
    mae = mean_absolute_error(test_output_actual, test_predictions)
    print(f"Mean Absolute Error: {mae}")

    # Calculate Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(test_output_actual, test_predictions)
    print(f"Mean Absolute Percentage Error: {mape}")

def create_Features_Scaling(hourly_df,num_input_days=72,num_forecast_hours=1):
    Year = []
    Month = []
    Day = []
    Hour = []
    Minute = []
    DayOfWeek = []
    Weekend = []
    Holiday = []
    TimeOfDay = []
    Summe = []
    Summe_ahead = []
    for index  in range(0,hourly_df.shape[0]-num_input_days-num_forecast_hours):
        
        # try:
            
            Year.append(hourly_df.loc[index:index+num_input_days,'Year'])
            Month.append(hourly_df.loc[index:index+num_input_days,'Month'])
            Day.append(hourly_df.loc[index:index+num_input_days,'Day'])
            Hour.append(hourly_df.loc[index:index+num_input_days,'Hour'])
            Minute.append(hourly_df.loc[index:index+num_input_days,'Minute'])
            DayOfWeek.append(hourly_df.loc[index:index+num_input_days,'DayOfWeek'])
            Weekend.append(hourly_df.loc[index:index+num_input_days,'Weekend'])
            Holiday.append(hourly_df.loc[index:index+num_input_days,'Holiday'])
            TimeOfDay.append(hourly_df.loc[index:index+num_input_days,'TimeOfDay'])
            Summe.append(hourly_df.loc[index:index+num_input_days,'Summe'])
            Summe_ahead.append(hourly_df.loc[index+num_input_days+num_forecast_hours,'Summe'])
        # except:
        #     continue

    Summe =  np.array(Summe)
    Year =  np.array(Year)
    Month =  np.array(Month)
    Day =  np.array(Day)
    Hour =  np.array(Hour)
    Minute =  np.array(Minute)
    DayOfWeek =  np.array(DayOfWeek)
    Weekend =  np.array(Weekend)
    Holiday =  np.array(Holiday)
    TimeOfDay =  np.array(TimeOfDay)

    # Target variables 
    Summe_ahead =  np.array(Summe_ahead)
    Summe_ahead = np.reshape(Summe_ahead,(len(Summe_ahead),num_forecast_hours))



    from sklearn.preprocessing import MinMaxScaler

    minMaxScaler = MinMaxScaler(feature_range=(0,1))

    Summe_scaler = minMaxScaler.fit_transform(Summe)
    Year_scaler = minMaxScaler.fit_transform(Year)
    Month_scaler = minMaxScaler.fit_transform(Month)
    Day_scaler = minMaxScaler.fit_transform(Day)
    Minute_scaler = minMaxScaler.fit_transform(Minute)
    DayOfWeek_scaler = minMaxScaler.fit_transform(DayOfWeek)
    Weekend_scaler = minMaxScaler.fit_transform(Weekend)
    Holiday_scaler = minMaxScaler.fit_transform(Holiday)
    TimeOfDay_scaler = minMaxScaler.fit_transform(TimeOfDay)
    Summe_ahead_scaler = minMaxScaler.fit_transform(Summe_ahead)
    
    X =  np.stack([Summe_scaler, Year_scaler,Month_scaler,Day_scaler,Minute_scaler,DayOfWeek_scaler,Weekend_scaler,Holiday_scaler,TimeOfDay_scaler],axis=2)
    Y =  Summe_ahead_scaler
    
    input_seq_train_val, input_seq_test, output_seq_train_val, output_seq_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    
    return input_seq_train_val, input_seq_test, output_seq_train_val, output_seq_test
if __name__=='__main__':
    
    
    # # loading the datasets from different files 
    path_lists = glob.glob('GreenD_reduced_version_03/'+'*.csv')
    sorted_file_paths = sorted(path_lists)
    sorted_file_paths = sorted(path_lists, key=extract_digits)
    
    conct_list = []
    for path in tqdm(sorted_file_paths ,desc='processing'):
        
        data = pd.read_csv(path)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'],utc=True,unit='s')#format='%Y-%m-%d %H-%M-%S')
        data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        conct_list.append(data)
        
    comnbineDf = pd.concat(conct_list)
    comnbineDf.sort_values(by='Timestamp',inplace=True)

    # os.makedirs('HourlyData',exist_ok=True)
  
    hourly_df = perform_downsampling(comnbineDf, freq='1H')
    # hourly_df.to_csv('HourlyData/hourlyDf.csv',index=False)
    
    # hourly_df =  pd.read_csv('HourlyData/hourlyDf.csv')
    hourly_df =  create_features(hourly_df,'Timestamp')
    hourly_df['TimeOfDay'] = hourly_df['TimeOfDay'].map({'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3})
    
    num_input_days = 24*3
    num_forecast_hours = 1
    timesteps_per_day = num_input_days * 24
    past_input_timesteps = num_input_days
    
     
    input_seq_train_val, input_seq_test, output_seq_train_val, output_seq_test= create_Features_Scaling(hourly_df,num_input_days,num_forecast_hours)
    train_input, val_input, train_output, val_output = train_test_split(input_seq_train_val, output_seq_train_val, test_size=0.25, shuffle=False)

    
    batch_size = 64
    epochs = 5

    steps_per_epoch = 200
    lstm_units = 200
    dense_units = 130
    learning_rate = 0.001
    input_shape = (train_input.shape[1] , train_input.shape[2])
    model = create_model(past_input_timesteps, lstm_units, dense_units, learning_rate,input_shape)
#     dump(model, 'ModelFiles/bidirectionalLstm.pkl')
        
    history, loss = train_and_evaluate_model(model, train_input, train_output, val_input, val_output, batch_size, epochs, steps_per_epoch)
    
    plot_loss(history)
# # 


#     # Load the saved scaler
#     scaler = load('ModelFiles/scaler.pkl')

    evaluate_model(model, test_input, test_output, scaler)
