import functools
import math
from typing import Callable

import numpy as np
import os
import statistics
from time import perf_counter_ns

import pandas as pd

from config import config_get

def function_execution_in_milliseconds(function_wrapper: Callable, *args, **kwargs):
    """
    A wrapper function for returning the elapsed runtime
    (in milliseconds) of a provided function along with the
    args of calling the function.
    :param function_wrapper: The function to call and time
    :param args: The arguments to pass to the function
    :return: (result of executing the desired function_wrapper,
            the execution time in nanoseconds)
    """
    start_time = perf_counter_ns()

    execution_result = function_wrapper(*args, **kwargs)

    stop_time = perf_counter_ns()
    elapsed_time = (stop_time - start_time) // 1_000_000

    return execution_result, elapsed_time


# gets the adjusted accuracy considering the accuracy of each model and the accuracy of the outliers (100%) adjusts based on amount
def get_recovered_accuracy(y_true: list, y_pred: list, threshold: float, metric="accuracy") -> float | None:
    if metric == "cosine":
        pass
    elif metric == "jaccard":
        pass
    else:
        percent_diff = list_of_percent_differences(y_true, y_pred)

        # Count pairs with percent difference below threshold
        below_threshold_count = sum(1 for diff in percent_diff if diff < threshold)

        # Calculate percentage of pairs below threshold
        accuracy = (below_threshold_count / len(y_true)) * 100

        return accuracy


def list_of_percent_differences(y_true: list, y_pred: list) -> list[float]:
    # Calculate percent difference for each pair of values
    percent_diff = []
    for true, pred in zip(y_true, y_pred):
        # Values are equal, handles 0/0 case
        if true == pred:
            diff = 0
        # Normal case
        elif true != 0:
            diff = abs((true - pred) / true) * 100
        # Case where predicted != 0 and true is. For our cases considered 100% difference
        else:
            diff = 100
        percent_diff.append(diff)

    return percent_diff


def percent_difference(y_true: list, y_pred: list) -> float:
    percent_diff = list_of_percent_differences(y_true, y_pred)

    # Calculate the average percent difference
    avg_percent_diff = sum(percent_diff) / len(percent_diff)

    return round(avg_percent_diff, 4)


def avg_cosine_similarity(y_true: list, y_pred: list) -> float:
    raise Exception("Cosine Similarity is not implemented in this version of the pipeline.")


def median_cosine_similarity(y_true: list, y_pred: list) -> float:
    raise Exception("Cosine Similarity is not implemented in this version of the pipeline.")

def minimum_cosine_similarity(y_true: list, y_pred: list) -> float:
    raise Exception("Cosine Similarity is not implemented in this version of the pipeline.")


def mse_metrics(y_true: list, y_pred: list, error_threshold=None) -> tuple[list[float], float, float, float, int] | tuple[list[float], float, float, float, None]:
    """
    A helper method for returning the mean-squared error (MSE)
    metrics such as the errors for each row, uniform average
    across all rows, standard deviation, variance, and if a
    threshold is provided, the number of rows with approximation error
    larger than the threshold (-1 if threshold is set to None)
    :param y_true: The array-like shape of ground truth values
    :param y_pred: The array-like shape of estimated target values
    :param error_threshold: Error threshold value, default of None for ignore
    :return: A tuple of the form (list of errors for each row,
            average, standard deviation, variance,
            number of rows whose approximated error exceeded error threshold)
    """
    sum_func = lambda a, b: a + b

    # Compute squared error and MSE of each value pair
    error_lst = list(map(lambda x, y: (y - x) ** 2, y_true, y_pred))
    aggregate_error_total = functools.reduce(sum_func, error_lst)
    mse = aggregate_error_total / len(y_true)

    # Compute Std Dev and Variance of squared error
    std_dev_numer = [(x - mse) ** 2 for x in error_lst]
    std_dev_sum = functools.reduce(sum_func, std_dev_numer)

    error_std_dev = math.sqrt(std_dev_sum / len(y_true))
    error_variance = error_std_dev ** 2

    if error_threshold is None:
        return error_lst, mse, error_std_dev, error_variance, None

    # Compute number of rows that exceed error threshold
    error_tuples = list(map(lambda x, y: (x, y), y_true, y_pred))
    approx_error_lst = list(map(lambda x: round(abs((x[1] - x[0]) / x[0]) * 100, 6), error_tuples))
    rows_exceeding_threshold = list(filter(lambda x: x > error_threshold, approx_error_lst))

    return error_lst, mse, error_std_dev, error_variance, len(rows_exceeding_threshold)


def size(num_clusters: int, num_outliers: int, datatype: str, num_records: int) -> tuple[float, float, float]:
    ml_models = {
        "lstm": 220_000,  # Size to store lstm model
        "linear_regression": 16,  # Size to store linear_regression model
    }
    datatype_length = size = np.dtype(datatype).itemsize

    cost_per_record_in_bits = math.ceil(
        math.log2(num_clusters + 1))  # cost associated with storing clustering information
    total_bytes_for_clusters = (cost_per_record_in_bits * num_records) // 8

    total_for_storing_models = num_clusters * ml_models[config_get('machine_learning_model')]
    total_for_outliers = num_outliers * datatype_length

    original_size = datatype_length * num_records
    total_size = total_bytes_for_clusters + total_for_storing_models + total_for_outliers
    size_as_percentage = round((total_size / original_size) * 100, 4)

    return total_size, original_size, size_as_percentage


def function_execution_in_nanoseconds(function_wrapper: Callable, *args, **kwargs):
    """
    A wrapper function for returning the elapsed runtime
    (in nanoseconds) of a provided function along with the
    args of calling the function.
    :param function_wrapper: The function to call and time
    :param args: The arguments to pass to the function
    :return: (result of executing the desired function_wrapper,
            the execution time in nanoseconds)
    """
    start_time = perf_counter_ns()

    execution_result = function_wrapper(*args, **kwargs)

    stop_time = perf_counter_ns()
    elapsed_time = stop_time - start_time

    return execution_result, elapsed_time


if __name__ == '__main__':
    true_targets = np.full(10, 0.5)
    pred_targets = np.full(10, 0.55)
    mse_result, mse_time = function_execution_in_nanoseconds(mse_metrics, true_targets, pred_targets,
                                                             error_threshold=5.0)
    (mse_lst, mse_avg, mse_std_dev, mse_var, mse_count) = mse_result
    for i in range(0, 10):
        print(f"Ground: {true_targets[i]}, Estimate: {pred_targets[i]}, Squared Error: {mse_lst[i]}\n")

    print(f"Average MSE: {mse_avg}\n"
          f"Standard Deviation: {mse_std_dev}\n"
          f"Variance: {mse_var}\n"
          f"Errors above threshold: {mse_count}\n"
          f"Execution Time: {mse_time} ns\n")
