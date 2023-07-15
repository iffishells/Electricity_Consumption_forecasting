import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
from sklearn.model_selection import train_test_split

from joblib import dump, load
def create_Features_Scaling(hourly_df,num_input_hours=72,num_forecast_hours=1):

    Year = []
    Month = []
    Day = []
    Hour = []
    DayOfWeek = []
    Weekend = []
    Holiday = []
    TimeOfDay = []
    Summe = []
    Summe_ahead = []
    
    for index  in range(0,hourly_df.shape[0]-num_input_hours-num_forecast_hours):
        Year.append(hourly_df.loc[index:index+num_input_hours-1,'Year'])
        Month.append(hourly_df.loc[index:index+num_input_hours-1,'Month'])
        Day.append(hourly_df.loc[index:index+num_input_hours-1,'Day'])
        Hour.append(hourly_df.loc[index:index+num_input_hours-1,'Hour'])
        DayOfWeek.append(hourly_df.loc[index:index+num_input_hours-1,'DayOfWeek'])
        Weekend.append(hourly_df.loc[index:index+num_input_hours-1,'Weekend'])
        Holiday.append(hourly_df.loc[index:index+num_input_hours-1,'Holiday'])
        TimeOfDay.append(hourly_df.loc[index:index+num_input_hours-1,'TimeOfDay'])
        Summe.append(hourly_df.loc[index:index+num_input_hours-1,'Summe'])
        Summe_ahead.append(hourly_df.loc[index+num_input_hours:index+num_input_hours+num_forecast_hours-1,'Summe'])

    Summe =  np.array(Summe)
    Year =  np.array(Year)
    Month =  np.array(Month)
    Day =  np.array(Day)
    Hour =  np.array(Hour)
    DayOfWeek =  np.array(DayOfWeek)
    Weekend =  np.array(Weekend)
    Holiday =  np.array(Holiday)
    TimeOfDay =  np.array(TimeOfDay)

    # Assuming all Summe_ahead entries have same length, equal to num_forecast_hours
    Summe_ahead =  np.array(Summe_ahead)

    minMaxScaler = MinMaxScaler(feature_range=(0,1))

    Summe_scaler = minMaxScaler.fit_transform(Summe)
    Year_scaler = minMaxScaler.fit_transform(Year)
    Month_scaler = minMaxScaler.fit_transform(Month)
    Day_scaler = minMaxScaler.fit_transform(Day)
    DayOfWeek_scaler = minMaxScaler.fit_transform(DayOfWeek)
    Weekend_scaler = minMaxScaler.fit_transform(Weekend)
    Holiday_scaler = minMaxScaler.fit_transform(Holiday)
    TimeOfDay_scaler = minMaxScaler.fit_transform(TimeOfDay)
    Summe_ahead_scaler = minMaxScaler.fit_transform(Summe_ahead)

        
    # Get the current date and time
    current_time = datetime.now()

    # Format the current time
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Print the formatted time
    dump(minMaxScaler, f'../Config_files/{formatted_time}_scaler.pkl')

    X =  np.stack([Summe_scaler, Year_scaler,Month_scaler,Day_scaler,DayOfWeek_scaler,Weekend_scaler,Holiday_scaler,TimeOfDay_scaler],axis=2)
    Y =  Summe_ahead_scaler

    input_seq_train_val, input_seq_test, output_seq_train_val, output_seq_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    
    return input_seq_train_val, input_seq_test, output_seq_train_val, output_seq_test
