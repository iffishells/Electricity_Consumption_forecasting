import glob
import re
import pandas as pd
from tqdm import tqdm
import os


def extract_digits(string):
    """
    Extracts the digits from a string using regular expression.

    Args:
        string (str): The input string.

    Returns:
        int: The extracted digits as an integer.

    """
    digits = re.findall(r'\d+', string)
    return int(digits[0]) if digits else 0


def combine_csv_files(folder_path, output_folder, saveFile=True):
    """
    Combines multiple CSV files in a folder into a single DataFrame.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
        output_folder (str): The path to the folder where the combined CSV file will be saved.
        saveFile (bool, optional): Whether to save the combined DataFrame as a CSV file.
                                   Defaults to True.

    Returns:
        pandas.DataFrame: The combined DataFrame.

    """
    path_lists = glob.glob(folder_path + '*.csv')
    sorted_file_paths = sorted(path_lists, key=lambda x: extract_digits(x))

    conct_list = []
    for path in tqdm(sorted_file_paths, desc='Processing'):
        data = pd.read_csv(path)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], utc=True, unit='s')
        data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        conct_list.append(data)

    combined_data = pd.concat(conct_list)
    combined_data.sort_values(by='Timestamp', inplace=True)

    if saveFile:
        output_path = output_folder
        os.makedirs(output_path, exist_ok=True)

        combined_data.to_csv(output_path + 'combined.csv', index=False)
        print("Files combined successfully.")

    return combined_data


def perform_downsampling(data, freq, aggregation_func='sum'):
    """
    Downsamples a DataFrame to a specified frequency and applies an aggregation function.

    Args:
        data (pandas.DataFrame): The input DataFrame.
        freq (str): The frequency to downsample to, e.g., '1H' for 1 hour, '30T' for 30 minutes.
        aggregation_func (str or function, optional): The aggregation function to apply during downsampling.
                                                      Defaults to 'sum'.

    Returns:
        pandas.DataFrame: The downsampled DataFrame.

    """
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
