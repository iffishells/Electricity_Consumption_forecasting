import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import holidays

# Define a function to check if a given date is a holiday
def is_holiday(date):
    at_holidays = holidays.Austria(years=date.year)
    return date in at_holidays

def creating_more_features(df, timestamp_col):
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
    df['DayOfWeek'] = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6
    df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # 1 if the day is weekend

    # Consider holidays if your data has specific trends on holidays


    # Apply the function to create a new 'Holiday' column
    df['Holiday'] = df['Timestamp'].apply(is_holiday).astype(int)


    # You may also consider to create a feature that represents the time of the day (morning, afternoon, evening, night)
    df['TimeOfDay'] = pd.cut(df['Hour'], bins=[-0.1, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    return df


# features_df['Timestamp'] = pd.to_datetime(features_df['Timestamp'])



