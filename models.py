import numpy as np
import pandas as pd
import os
import warnings
import statistics

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LinearRegression

import models
from config import config_get
from pathlib import Path

from helper import list_of_percent_differences, list_of_cosine_similarities

column_index_variable = config_get('index_column_name')


class Cluster:
    def __init__(self, model, inliers: list[int], original_y_values: list[float], predicted_y_values: list[float], value=None, predicting_feature_count=1, representative_point:list=None, cluster_index=0):
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
        self.cluster_index = cluster_index + 1

        self.size = 0
        self.original_y_values = original_y_values
        self.predicted_y_values = predicted_y_values
        self.predicting_feature_count = predicting_feature_count

        if representative_point is None:
            predictor_array = [0 for _ in range(predicting_feature_count)]

            if hasattr(model, "input_shape"):
                rep = np.array(predictor_array, dtype=np.float32).reshape(1, 1, predicting_feature_count)
            else:
                rep = np.array(predictor_array, dtype=np.float32).reshape(1, -1)

            self.sample_y_value = model.predict(rep)
            self.representative_point = None
            return

        if not isinstance(representative_point, list):
            representative_point = representative_point.tolist()

        rep = np.array(representative_point, dtype=np.float64)

        if hasattr(model, "input_shape"):
            # tensorflow and keras "fun"
            if rep.ndim == 0:
                rep = rep.reshape(1, 1, 1)
            elif rep.ndim == 1:
                rep = rep.reshape(1, 1, rep.shape[0])
            elif rep.ndim == 2:
                rep = rep.reshape(1, rep.shape[0], rep.shape[1])
            else:
                raise ValueError(f"Unexpected representative_point shape: {rep.shape}")
            self.sample_y_value = model.predict(rep)
        else:
            if not isinstance(representative_point, list):
                representative_point = representative_point.tolist()

            self.sample_y_value = model.predict(np.array(representative_point).reshape(1, -1))

        if self.sample_y_value.size == 1:
            y_scalar = float(self.sample_y_value)
            self.representative_point = representative_point + [y_scalar]
        else:
            y_vector = self.sample_y_value.reshape(-1).tolist()
            self.representative_point = representative_point + y_vector
            # if rep.ndim == 0:
            #     rep = rep.reshape(1, 1)
            # elif rep.ndim == 1:
            #     rep = rep.reshape(1, rep.shape[0])
            # else:
            #     rep = np.mean(rep, axis=0).reshape(1, -1)

        # print(len(self.representative_point))
        # self.representative_point = representative_point

    def length(self):
        """
        Return the length of the inliers list.
        """
        if len(self.inliers) > 0:
            return len(self.inliers)

        return self.size

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
        self.size = len(self.inliers)

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


def train_and_evaluate_model(data: pd.DataFrame, x_label: str, y_label: str, percent_acceptable: float, metric="accuracy", cluster_index=0, predicting_feature_count=1, representative_point=None) -> tuple[models.Cluster, list]:
    """
    Trains a model on the provided `data` and evaluates its performance. 
    Returns a Cluster object containing inliers, representing data points predicted within a specified threshold.
    
    Args:
        - data (DataFrame): The dataset containing strs and target variable.
        - x_label (str): The label of the str to be used as input for training the model.
        - y_label (str): The label of the target variable.
        - percent_acceptable (float): The acceptable percentage deviation from the true values for inliers.
        
    Returns:
        - cluster (Cluster): A Cluster object containing inliers predicted by the model. None is returned if all values
            are marked as outliers (i.e., no inliers are detected due to error, model choice, etc.)
        - outliers (list of int): A list of indices corresponding to outliers predicted by the model.
    """

    # Train the linear regression model and get the predictions
    model, y_pred = run_model(data, x_label, y_label)

    # Add y_pred to the stripped table
    stripped_table = data[[column_index_variable, y_label]].copy()
    stripped_table["y_pred"] = list(y_pred)

    # Calculate absolute percentage error
    y = list(np.array(data[y_label]))
    if metric == "cosine":
        abs_cosine_similarity = list_of_cosine_similarities(y, list(y_pred))
        stripped_table["abs_cosine_similarity"] = abs_cosine_similarity
    else:
        abs_percentage_error = list_of_percent_differences(y, list(y_pred))
        stripped_table["abs_percentage_error"] = abs_percentage_error

    if "abs_cosine_similarity" in stripped_table:
        inliers_table = stripped_table[stripped_table["abs_cosine_similarity"] >= percent_acceptable]
        outliers_table = stripped_table[stripped_table["abs_cosine_similarity"] < percent_acceptable]
    else:
        # Filter rows to get inliers and outliers table
        inliers_table = stripped_table[stripped_table["abs_percentage_error"] <= percent_acceptable]
        outliers_table = stripped_table[stripped_table["abs_percentage_error"] > percent_acceptable]

    # Get inliers and outliers as lists
    inliers = inliers_table[column_index_variable].tolist()
    outliers = outliers_table[column_index_variable].tolist()

    # Get original and predicted values from the inliers table
    inliers_original_y = inliers_table[y_label].tolist()
    inliers_predicted_y = inliers_table["y_pred"].tolist()

    # Determine representative point for the new cluster based on inliers using median
    inlier_data_points = data[data[column_index_variable].isin(inliers)]

    if not hasattr(data[x_label].iloc[0], '__len__'):
        if len(inlier_data_points) > 0:
            representative_point_lst = [statistics.median(inlier_data_points[x_label])]
        else:
            representative_point_lst = [0]

        representative_point = representative_point_lst

    # Create a Cluster object with additional fields for original and predicted y values
    cluster = Cluster(model, inliers, inliers_original_y, inliers_predicted_y, cluster_index=cluster_index, predicting_feature_count=predicting_feature_count, representative_point=representative_point)

    return cluster, outliers


def run_model(data: pd.DataFrame, x_label: str, y_label: str):
    models = {
        "lstm": create_lstm,
        "linear_regression": create_linear_regression
    }
    model_function = config_get("machine_learning_model")

    # Check if one global model type is used or if each column is specified
    use_one_model_type = config_get("use_one_model_type")
    if not use_one_model_type:
        predictors_dict = config_get("predicted_by")
        model_function = predictors_dict[y_label]['model']
    # Get the corresponding function based on the model_type
    model_func = models.get(model_function)
    if model_func:
        return model_func(data, x_label, y_label)
    else:
        return "Unknown model type"


def multivariable_train_and_evaluate_model(data: pd.DataFrame, x_label: list[str], y_label: str, percent_acceptable: float,
                             metric="accuracy", cluster_index=0) -> tuple[models.Cluster, list]:
    """
    Trains a model on the provided `data` and evaluates its performance.
    Returns a Cluster object containing inliers, representing data points predicted within a specified threshold.

    Args:
        - data (DataFrame): The dataset containing strs and target variable.
        - x_label (str): The label of the str to be used as input for training the model.
        - y_label (str): The label of the target variable.
        - percent_acceptable (float): The acceptable percentage deviation from the true values for inliers.

    Returns:
        - cluster (Cluster): A Cluster object containing inliers predicted by the model. None is returned if all values
            are marked as outliers (i.e., no inliers are detected due to error, model choice, etc.)
        - outliers (list of int): A list of indices corresponding to outliers predicted by the model.
    """

    # Train the linear regression model and get the predictions
    model, y_pred = multivariable_run_model(data, x_label, y_label)

    # Add y_pred to the stripped table
    stripped_table = data[[column_index_variable, y_label]].copy()
    stripped_table["y_pred"] = list(y_pred)

    # Calculate absolute percentage error
    y = list(np.array(data[y_label]))
    if metric == "cosine":
        abs_cosine_similarity = list_of_cosine_similarities(y, list(y_pred))
        stripped_table["abs_cosine_similarity"] = abs_cosine_similarity
    else:
        abs_percentage_error = list_of_percent_differences(y, list(y_pred))
        stripped_table["abs_percentage_error"] = abs_percentage_error

    if "abs_cosine_similarity" in stripped_table:
        inliers_table = stripped_table[stripped_table["abs_cosine_similarity"] >= percent_acceptable]
        outliers_table = stripped_table[stripped_table["abs_cosine_similarity"] < percent_acceptable]
    else:
        # Filter rows to get inliers and outliers table
        inliers_table = stripped_table[stripped_table["abs_percentage_error"] <= percent_acceptable]
        outliers_table = stripped_table[stripped_table["abs_percentage_error"] > percent_acceptable]

    # Get inliers and outliers as lists
    inliers = inliers_table[column_index_variable].tolist()
    outliers = outliers_table[column_index_variable].tolist()

    # Get original and predicted values from the inliers table
    inliers_original_y = inliers_table[y_label].tolist()
    inliers_predicted_y = inliers_table["y_pred"].tolist()

    # Determine representative point for the new cluster based on inliers using median
    inlier_data_points = data[data[column_index_variable].isin(inliers)]
    # print(f"{inlier_data_points})")
    # print(f"{len(inlier_data_points)}")

    if len(inlier_data_points) > 0:
        representative_point_lst = [statistics.median(inlier_data_points[cur_x_lab]) for cur_x_lab in x_label]
    else:
        representative_point_lst = [0 for _ in x_label]

    # Create a Cluster object with additional fields for original and predicted y values
    cluster = Cluster(model, inliers, inliers_original_y, inliers_predicted_y, predicting_feature_count=len(x_label), representative_point=representative_point_lst, cluster_index=cluster_index)

    return cluster, outliers


def multivariable_run_model(data: pd.DataFrame, x_label: list[str], y_label: str):
    models = {
        "multivariable LR": create_multiple_regression
    }
    model_function = config_get("machine_learning_model")

    # Check if one global model type is used or if each column is specified
    use_one_model_type = config_get("use_one_model_type")
    if not use_one_model_type:
        predictors_dict = config_get("predicted_by")
        model_function = predictors_dict[y_label]['model']
    # Get the corresponding function based on the model_type
    model_func = models.get(model_function)
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
    x_raw = data[x_label].values
    y_raw = data[y_label].values

    def detect_structure(val):
        if not hasattr(val, '__len__'):
            return "scalar"
        if hasattr(val, '__len__') and not hasattr(val[0], '__len__'):
            return "vector"
        return "sequence"

    X_type = detect_structure(x_raw[0])
    Y_type = detect_structure(y_raw[0])

    # Prepare X
    if X_type == "scalar":
        X = x_raw.astype(float).reshape(-1, 1)
        X = (X - X.mean()) / (X.std() or 1)
        X = X.reshape(len(X), 1, 1)

    elif X_type == "vector":
        X = np.array(list(x_raw), dtype=np.float32)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)
        X = (X - mean) / std
        X = X.reshape(len(X), 1, X.shape[1])

    else:  # sequence
        X = pad_sequences(x_raw, dtype='float32', padding='post')
        mean = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True)
        std = np.where(std == 0, 1, std)
        X = (X - mean) / std

    # Prepare Y
    if Y_type == "scalar":
        y = y_raw.astype(float).reshape(-1, 1)
        y = (y - y.mean()) / (y.std() or 1)
        output_dim = 1

    elif Y_type == "vector":
        y = np.array(list(y_raw), dtype=np.float32)
        mean = y.mean(axis=0, keepdims=True)
        std = y.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)
        y = (y - mean) / std
        output_dim = y.shape[1]

    else:  # sequence
        y = pad_sequences(y_raw, dtype='float32', padding='post')
        mean = y.mean(axis=(0, 1), keepdims=True)
        std = y.std(axis=(0, 1), keepdims=True)
        std = np.where(std == 0, 1, std)
        y = (y - mean) / std
        output_dim = y.shape[2]

    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(64))
    model.add(Dense(output_dim))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=1, batch_size=32, validation_data=(X, y))

    predictions = model.predict(X)

    # Safe denormalization
    std_y = y_raw.std()
    mean_y = y_raw.mean()
    std_y_safe = np.where(std_y == 0, 1, std_y)
    predictions = predictions * std_y_safe + mean_y

    if Y_type == "scalar":
        y_pred = predictions.flatten()
    else:
        y_pred = predictions

    return model, y_pred


def create_linear_regression(data: pd.DataFrame, x_label: str, y_label: str) -> tuple:
    x, y = extract_xy_columns(data, x_label, y_label)

    # Initialize Linear Regression model
    # x = np.array(x).reshape(-1, 1)
    model = LinearRegression().fit(x, y)

    # predict y values based on input x
    y_pred = model.predict(x)

    return model, y_pred


def create_multiple_regression(data: pd.DataFrame, x_label: list[str], y_label: str) -> tuple:
    # x, y = extract_xy_columns(data, x_label, y_label)
    x = data[x_label].to_numpy()
    _, y = extract_xy_columns(data, x_label[0], y_label)

    # Initialize Linear Regression model
    model = LinearRegression().fit(x, y)

    # predict y values based on input x
    y_pred = model.predict(x)

    return model, y_pred
