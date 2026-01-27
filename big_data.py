"""
This file includes the functionality to expand a database. It had two primary functions,
one which can expand a dataset while adding noise and a second that copies values and 
does not add noise. Currently only the first function is used in the pipeline.
"""
import pandas as pd
import numpy as np
from config import column_variable_get


def data_big_with_noise(data: pd.DataFrame, multiplier: int, expanded_file_name: str) -> pd.DataFrame:
    """
    Expand data by copying it multiplier times. Uses concatenation for quicker run time.
    Adds noise to columns with continuous data as defined in `configuration files/datasets_info.yaml`
    Save new dataset to `expanded_file_name`
    """

    # Function to add noise to continuous columns
    def add_noise_to_continuous_column(column, noise_level=3):
        noise = np.random.normal(0, scale=noise_level, size=len(column))
        noisy_column = column + noise
        noisy_column = noisy_column.astype(int)
        return noisy_column

    original_length = len(data)

    # Create a list of copies and concatenate
    dfs = [data.copy() for _ in range(multiplier)]
    expanded_data = pd.concat(dfs, ignore_index=True)

    # Get column variable types (continuous or categorical)
    for column_name in expanded_data.columns:
        column_type = column_variable_get(column_name)

        if column_type == 'continuous':
            # Add noise only to the new rows
            expanded_data.loc[original_length:, column_name] = add_noise_to_continuous_column(
                expanded_data.loc[original_length:, column_name])

    # Reset index for all rows
    expanded_data['Index'] = range(1, len(expanded_data) + 1)

    # Save file to given new name
    if expanded_file_name is not None:
        expanded_data.to_csv(expanded_file_name, index=False)

    return expanded_data


def data_big(data: pd.DataFrame, multiplier: int, expanded_file_name: str) -> pd.DataFrame:
    """
    Copy data multiplierX times and save to expanded_file_name
    """
    dfs = [data] * multiplier
    expanded_data = pd.concat(dfs, ignore_index=True)

    # Reset the 'Index' column starting from max_index + 1
    max_index = max(expanded_data['Index']) + 1
    expanded_data['Index'] = range(max_index, max_index + len(expanded_data))

    # Save file to given name
    if expanded_file_name is not None:
        expanded_data.to_csv(expanded_file_name, index=False)

    return expanded_data


def shrink_file(data: pd.DataFrame, desired_number_of_records: int, destination: str) -> None:
    """
    Takes in a pandas DataFrame `data` and writes out the first `desired_number_of_records`
    into a new CSV file given by file path + name `destination`.

    Parameters:
        data (pd.DataFrame): The DataFrame to shrink.
        desired_number_of_records (int): The number of records from the start of the DataFrame to write to the CSV.
        destination (str): The file path and name where the CSV should be saved.
    """
    if desired_number_of_records > len(data):
        raise ValueError("Desired number of records exceeds the available records in the DataFrame.")

    # Select the first `desired_number_of_records` from the DataFrame and write to CSV
    truncated_data = data.head(desired_number_of_records)
    truncated_data.to_csv(destination, index=False)

    print("reduced size of file")


def main():
    data = pd.read_csv('archive/sntemp_inputs_outputs_lordville.csv', sep=',')
    shrink_file(data, 20000, 'archive/sntemp_inputs_outputs_lordville_mini.csv')


if __name__ == "__main__":
    main()
