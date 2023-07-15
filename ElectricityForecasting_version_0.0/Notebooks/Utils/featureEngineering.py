import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
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
    cal = calendar()
    holidays = cal.holidays(start=df[timestamp_col].min(), end=df[timestamp_col].max())
    df['Holiday'] = df[timestamp_col].isin(holidays).astype(int)  # 1 if the day is a US Federal holiday

    # You may also consider to create a feature that represents the time of the day (morning, afternoon, evening, night)
    df['TimeOfDay'] = pd.cut(df['Hour'], bins=[-0.1, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    return df