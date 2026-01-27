import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression

import models
from config import config_get

from helper import list_of_percent_differences

MODEL_FUNCTION = config_get("machine_learning_model")


class Cluster:
    def __init__(self, model, inliers: list[int], original_y_values: list[float], predicted_y_values: list[float], value=None):
        """
        Initialize a Cluster object with model,inliers, original y values, and predicted y values. Size represents the number of values that have
        been flushed from a cluster (i.e. cleared out from inliers/original_y_values/predicted_y_values for memory purposes)

        Args:
        - model (model): The model associated with the cluster.
        - inliers (list of int): List of indices corresponding to inliers in the cluster.
        - original_y_values (list of float): Original y values for inliers (optional).
        - predicted_y_values (list of float): Predicted y values for inliers (optional).
        """
        self.model = model
        self.value = value
        self.inliers = inliers
        self.size = 0
        self.original_y_values = original_y_values
        self.predicted_y_values = predicted_y_values
        self.sample_y_value = model.predict(np.array([[0]]))

    def length(self):
        """
        Return the length of the inliers list.
        """
        return len(self.inliers)

    def __lt__(self, other):
        """
        Less than comparison based on the length of inliers.
        """
        return self.length() < other.length()

    def compare_by_predicted_y_value(self, other):
        return self.sample_y_value < other.sample_y_value

    def __str__(self):
        """
        Return a string representation of the Cluster object.
        """
        return f"Cluster Model: {self.model}, Length: {self.length()}, Original:\n{self.original_y_values}, Predicted:\n{self.predicted_y_values}"

    def add_new_value(self, index=None, original_y_value=None, predicted_y_value=None):
        """
        Add a new value to the cluster (done as a result batching process)
        """
        if index:
            self.inliers.append(index)
        else:
            self.size += 1
        if original_y_value:
            self.original_y_values.append(original_y_value)
        if predicted_y_value:
            self.predicted_y_values.append(predicted_y_value)

    def flush_cluster(self, destination=None, predicted_label=None, cur_partition_num=0):
        """
        Clears out original_y_values and predicted_y_values and inliers. This is for large datasets that can't store all of this
        information in memory
        """
        self.size += len(self.inliers)

        if destination is not None:
            inlier_csv_path = destination.parent / f'{predicted_label}' / f'partition-{cur_partition_num}.csv'
            if not os.path.exists(inlier_csv_path):
                os.makedirs(os.path.dirname(inlier_csv_path), exist_ok=True)

            buffer_dataframe = pd.DataFrame()
            buffer_dataframe['inliers'] = self.inliers
            buffer_dataframe['original_y_values'] = self.original_y_values
            buffer_dataframe['predicted_y_values'] = self.predicted_y_values
            buffer_dataframe.to_csv(inlier_csv_path, index=False, header=False, mode='a')

        self.inliers = []
        self.original_y_values = []
        self.predicted_y_values = []



def train_and_evaluate_model(data: pd.DataFrame, x_label: str, y_label: str, percent_acceptable: float, metric="accuracy") -> tuple[models.Cluster, list]:
    """
    Trains a model on the provided `data` and evaluates its performance. 
    Returns a Cluster object containing inliers, representing data points predicted within a specified threshold.
    
    Args:
        - data (DataFrame): The dataset containing strs and target variable.
        - x_label (str): The label of the str to be used as input for training the model.
        - y_label (str): The label of the target variable.
        - percent_acceptable (float): The acceptable percentage deviation from the true values for inliers.
        
    Returns:
        - cluster (Cluster): A Cluster object containing inliers predicted by the model.
        - outliers (list of int): A list of indices corresponding to outliers predicted by the model.
    """

    # Train the linear regression model and get the predictions
    model, y_pred = run_model(data, x_label, y_label)

    # Add y_pred to the stripped table
    stripped_table = data[["Index", y_label]].copy()
    stripped_table["y_pred"] = list(y_pred)

    # Calculate absolute percentage error
    y = list(np.array(data[y_label]))
    if metric == "cosine":
        raise Exception("Cosine similarity accuracy metric currently not implemented")
    elif metric == "jaccard":
        raise Exception("Jaccard Similarity accuracy metric currently not implemented")
    else:
        abs_percentage_error = list_of_percent_differences(y, list(y_pred))
        stripped_table["abs_percentage_error"] = abs_percentage_error

    if "abs_cosine_similarity" in stripped_table:
        inliers_table = stripped_table[stripped_table["abs_cosine_similarity"] >= percent_acceptable]
        outliers_table = stripped_table[stripped_table["abs_cosine_similarity"] < percent_acceptable]
    elif "abs_jaccard_similarity" in stripped_table:
        pass
    else:
        # Filter rows to get inliers and outliers table
        inliers_table = stripped_table[stripped_table["abs_percentage_error"] <= percent_acceptable]
        outliers_table = stripped_table[stripped_table["abs_percentage_error"] > percent_acceptable]

    # Get inliers and outliers as lists
    inliers = inliers_table["Index"].tolist()
    outliers = outliers_table["Index"].tolist()

    # Get original and predicted values from the inliers table
    inliers_original_y = inliers_table[y_label].tolist()
    inliers_predicted_y = inliers_table["y_pred"].tolist()

    # Create a Cluster object with additional fields for original and predicted y values
    cluster = Cluster(model, inliers, inliers_original_y, inliers_predicted_y)

    return cluster, outliers


def run_model(data: pd.DataFrame, x_label: str, y_label: str):
    models = {
        "lstm": create_lstm,
        "linear_regression": create_linear_regression
    }
    # Get the corresponding function based on the model_type
    model_func = models.get(MODEL_FUNCTION)
    if model_func:
        return model_func(data, x_label, y_label)
    else:
        return "Unknown model type"


def extract_xy_columns(data: pd.DataFrame, x_label: str, y_label):
    if pd.api.types.is_numeric_dtype(data[x_label]):
        if isinstance(np.array(data[x_label])[0], np.ndarray):
            x = np.array([np.ravel(arr) for arr in data[x_label]])
        else:
            x = np.array(data[x_label]).reshape(-1, 1)  # Ensure x is 2D

        # Prepare y
        if isinstance(np.array(data[y_label])[0], np.ndarray):
            y = np.array([np.ravel(arr) for arr in data[y_label]]).flatten()
        else:
            y = np.array(data[y_label]).flatten()  # Ensure y is 1D
    else:
        if isinstance(np.array(data[x_label])[0], np.ndarray):
            x = [np.ravel(arr) for arr in data[x_label]]
        elif isinstance(np.array(data[x_label]), np.ndarray):
            x = [arr for arr in data[x_label]]
        else:
            x = np.array(data[x_label]).reshape((-1, 1))

        # Prepare y
        if isinstance(np.array(data[y_label])[0], np.ndarray):
            y = [np.ravel(arr) for arr in data[y_label]]
        else:
            y = np.array(data[y_label])

    return x, y


def create_lstm(data: pd.DataFrame, x_label: str, y_label: str) -> tuple:
    # Convert data to numpy arrays
    original_x_values = data[x_label].values.reshape(-1, 1)
    original_y_values = data[y_label].values.reshape(-1, 1)

    # normalizing the input age data ensures that all age values are scaled to the range [0,1]
    normalized_x = (original_x_values - np.mean(original_x_values)) / np.std(original_x_values)
    normalized_y = (original_y_values - np.mean(original_y_values)) / np.std(original_y_values)

    X = np.array(normalized_x)
    y = np.array(normalized_y)

    # Define LSTM model
    model = Sequential()
    sequence_length = 1
    model.add(
        LSTM(64, input_shape=(sequence_length, 1)))  # this 1 represents the number of strs. age is the only one
    model.add(Dense(1))

    # Compile and train model
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=1, batch_size=32, validation_data=(X, y))

    # Normalized predictions
    predictions = model.predict(X)

    # Denormalize predictions and actual value
    predictions_denorm = (predictions * np.std(normalized_y)) + np.mean(normalized_y)
    y_pred = (predictions_denorm * np.std(original_y_values)) + np.mean(original_y_values)
    y_pred = y_pred.flatten()

    return model, y_pred


def create_linear_regression(data: pd.DataFrame, x_label: str, y_label: str) -> tuple:
    x, y = extract_xy_columns(data, x_label, y_label)

    # Initialize Linear Regression model
    model = LinearRegression().fit(x, y)

    # predict y values based on input x
    y_pred = model.predict(x)

    return model, y_pred
