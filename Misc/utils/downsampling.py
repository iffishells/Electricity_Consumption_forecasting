import pandas as pd
import numpy as np
# The code defines a function named downsample_dataframe that takes in three parameters: df (the DataFrame to be downsampled), 
# downsampling_frequency (the desired frequency to downsample the DataFrame to), 
# and fill_method (the method to use for filling missing values).
# The function aims to downsample the DataFrame and fill any missing value
def downsample_dataframe(df, downsampling_frequency, fill_method='mean'):
  """
  Downsamples a DataFrame and fills missing values.

  Args:
    df: The DataFrame to downsample.
    downsampling_frequency: The frequency to downsample the DataFrame to.
    fill_method: The method to use to fill missing values.

  Returns:
    The downsampled DataFrame.
  """

  # Convert the Timestamp column to a datetime object.
  # This line converts the 'Timestamp' column in the DataFrame to a datetime object using the pd.to_datetime() function from the pandas library.
  df['Timestamp'] = pd.to_datetime(df['Timestamp'])

  # Set the index of the DataFrame to the Timestamp column.
  # This line sets the 'Timestamp' column as the index of the DataFrame using the set_index() method
  df = df.set_index('Timestamp')

  # Fill the missing values.
  # This block of code fills the missing values in the DataFrame based on the specified fill_method. 
  # If fill_method is set to 'mean', the missing values are filled with the mean value of each 
  # column using the fillna() method and df.mean(). If fill_method is set to 'median', 
  # he missing values are filled with the median value of each column using df.median(). 
  # If an invalid fill_method is provided, a ValueError is raised
  if fill_method == 'mean':
    df = df.fillna(df.mean())
  elif fill_method == 'median':
    df = df.fillna(df.median())
  else:
    raise ValueError('Invalid fill_method: {}'.format(fill_method))

  # Downsample the DataFrame.
  # This line downsamples the DataFrame to the specified downsampling_frequency using the resample() method with 
  # downsampling_frequency as the argument. The mean() method is then applied to calculate the mean 
  # value for each downsampled interval. The resulting downsampled DataFrame is returned as the output of the function.
  df = df.resample(downsampling_frequency).mean()
  return df

# Summary:
# The code defines a function downsample_dataframe() that takes in a DataFrame, downsampling frequency, 
# and fill method. The function converts the 'Timestamp' column to a datetime object, sets it as the index, 
# fills missing values based on the specified fill method (mean or median), and then downsamples the DataFrame 
# to the desired frequency using the mean value within each downsampled interval. 
# The resulting downsampled DataFrame is returned as the output of the function


