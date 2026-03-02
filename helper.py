import functools
import math
from typing import Callable

import numpy as np
import os
import statistics
from time import perf_counter_ns
import struct
import heapq
import shutil

import pandas as pd
from pathlib import Path
import dask.dataframe as dd

from config import config_get

from text import cosine_similarity


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


def create_file_if_not_exists(file_path: str) -> None:
    """
    Creates a file and its directory if it doesn't exist. Useful for results.
    """
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)


def format_duration(ms: int) -> str:
    # If it's small, keep ms
    if ms < 1000:
        return f"{ms} ms"

    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0:
        parts.append(f"{seconds}s")
    if ms > 0:
        parts.append(f"{ms}ms")

    return " ".join(parts)


def parse_byte_size(s: str) -> int:
    s = s.strip().upper()

    # Split into numeric part + unit part
    num = ""
    unit = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            unit += ch

    num = float(num)

    unit_multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    unit = unit.strip()
    if unit == "":
        multiplier = 1
    else:
        multiplier = unit_multipliers[unit]

    return int(num * multiplier)



def get_record_size(data):
    data_columns = list(data.columns.values)
    column_sizes = [data.dtypes[col].itemsize for col in data_columns]
    return np.sum(column_sizes)


def append_outlier_records(outliers, row_indices, attribute_index, datatype: str, file_path):
    """
    Append outlier records to a temp file.
    Each record = [row_index][attribute_index][value]
    """
    byte_width = np.dtype(datatype).itemsize
    float_fmt_map = {4: "f", 8: "d"}
    float_fmt = float_fmt_map[byte_width]

    # Full record format: row_index (I), attribute_index (H), value (float_fmt)
    fmt = f">I H {float_fmt}"

    with open(file_path, "ab") as f:
        for row_idx, val in zip(row_indices, outliers):
            row_idx = int(row_idx)
            attribute_index = int(attribute_index)
            val = float(val)

            f.write(struct.pack(fmt, row_idx, attribute_index, val))


def sort_temp_file_in_chunks(temp_path, record_size, chunk_size_bytes=5_000_000):

    # See how many records would fit within a chunk (rounded down) and use that to choose a chunk size accordingly
    records_per_chunk = chunk_size_bytes // record_size
    updated_chunk_size = record_size * records_per_chunk

    assert (updated_chunk_size % record_size) == 0

    chunks = []

    with open(temp_path, "rb") as f:
        while True:
            chunk = f.read(updated_chunk_size)
            if not chunk:
                break

            # Break chunk into records
            records = [
                chunk[i:i+record_size]
                for i in range(0, len(chunk), record_size)
            ]

            # Sort by (row_index, attribute_index)
            records.sort(key=lambda r: struct.unpack(">I H", r[:6]))

            # Write sorted chunk to a new file
            chunk_path = temp_path.with_suffix(f".chunk{len(chunks)}")
            with open(chunk_path, "wb") as cf:
                for rec in records:
                    cf.write(rec)

            chunks.append(chunk_path)

    return chunks


def load_cluster_map_files(cluster_map_root: Path) -> dd.DataFrame:
    """
    Load all per-attribute cluster mapping tables and merge them into a single
    wide-format Dask DataFrame keyed by row_index.
    Any missing (row_index, attribute) entry is filled with cluster_id = 0.

    Directory structure expected:
        cluster_map/
            col_012/
                part_00000.parquet
                part_00001.parquet
            col_013/
                part_00000.parquet
                ...
    """
    mapping_tables = []

    # Load each attribute directory as a Dask dataframe
    for attr_dir in cluster_map_root.iterdir():
        if attr_dir.is_dir():
            df = dd.read_parquet(str(attr_dir))

            # Ensure the attribute column exists and is integer
            attr_name = [c for c in df.columns if c != "row_index"][0]
            df[attr_name] = df[attr_name].astype("int32")

            mapping_tables.append(df)

    if not mapping_tables:
        raise ValueError(f"No mapping tables found in {cluster_map_root}")

    # Merge all mapping tables on row_index
    cluster_map = mapping_tables[0]
    for df in mapping_tables[1:]:
        cluster_map = cluster_map.merge(df, on="row_index", how="outer")

    # Fill missing cluster IDs with 0
    attr_cols = [c for c in cluster_map.columns if c != "row_index"]
    cluster_map[attr_cols] = cluster_map[attr_cols].fillna(0).astype("int32")

    return cluster_map


def merge_sorted_chunks(chunk_paths, output_path, record_size):
    files = [open(p, "rb") for p in chunk_paths]
    heap = []

    # Initialize heap with first record from each file
    for i, f in enumerate(files):
        rec = f.read(record_size)
        if rec:
            row_idx, attr_idx = struct.unpack(">I H", rec[:6])
            heapq.heappush(heap, (row_idx, attr_idx, rec, i))

    with open(output_path, "wb") as out:
        while heap:
            _, _, rec, file_idx = heapq.heappop(heap)
            out.write(rec)

            next_rec = files[file_idx].read(record_size)
            if next_rec:
                row_idx, attr_idx = struct.unpack(">I H", rec[:6])
                heapq.heappush(heap, (row_idx, attr_idx, next_rec, file_idx))

    for f in files:
        f.close()


def clear_temp_dir(temp_dir):
    """
    Remove the temporary outlier directory and all its contents.
    """
    if temp_dir.exists() and temp_dir.is_dir():
        shutil.rmtree(temp_dir)


def persist_cluster(clusters: list, path_to_write: Path, model_type: str, predictor_features: list[str], predicted_feature: str, batch_size: str):


    if model_type == 'linear_regression':
        model_columns = ['cluster_index', f'{predictor_features[0]}_slope', 'intercept']
        cluster_out_df = pd.DataFrame(columns=model_columns)
        cluster_out_df['cluster_index'] = [x.cluster_index for x in clusters]
        cluster_out_df[f'{predictor_features[0]}_slope'] = [x.model.coef_[0] for x in clusters]
        cluster_out_df['intercept'] = [x.model.intercept_ for x in clusters]
        cluster_out_df = cluster_out_df.sort_values(by=['cluster_index'])
        output_path = path_to_write / 'linear_regression' / f'{predicted_feature}'
        output_path.mkdir(parents=True, exist_ok=True)

        cluster_out_ddf = dd.from_pandas(cluster_out_df, chunksize=parse_byte_size(batch_size))
        cluster_out_ddf.to_parquet(output_path, engine='pyarrow', write_index=False)
    elif model_type == 'multivariable LR':
        raise ValueError("Multivariable LR not available in this version.")
    elif model_type == 'lstm':
        for cluster in clusters:
            output_path = path_to_write / 'lstm' / f'model_{cluster.cluster_index}'
            output_path.mkdir(parents=True, exist_ok=True)

            cluster.model.save(output_path)
    else:
        raise ValueError('Unsupported model type when serializing cluster')
    pass


# gets the adjusted accuracy considering the accuracy of each model and the accuracy of the outliers (100%) adjusts based on amount
def get_recovered_accuracy(y_true: list, y_pred: list, threshold: float, metric="accuracy") -> float | None:
    if metric == "cosine":
        cos_sim = list_of_cosine_similarities(y_true, y_pred)

        above_threshold_count = sum(1 for sim in cos_sim if sim > threshold)

        accuracy = (above_threshold_count / len(y_true)) * 100

        return accuracy
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


def list_of_cosine_similarities(y_true: list, y_pred: list) -> list[float]:
    cos_sim = []
    for true, pred in zip(y_true, y_pred):
        cos_sim.append(cosine_similarity(true, pred))
    return cos_sim


def percent_difference(y_true: list, y_pred: list) -> float:
    percent_diff = list_of_percent_differences(y_true, y_pred)

    # Calculate the average percent difference
    avg_percent_diff = sum(percent_diff) / len(percent_diff)

    return round(avg_percent_diff, 4)


def avg_cosine_similarity(y_true: list, y_pred: list) -> float:
    cos_sim = list_of_cosine_similarities(y_true, y_pred)

    avg_cos_sim = sum(cos_sim) / len(cos_sim)

    return round(avg_cos_sim, 4)


def median_cosine_similarity(y_true: list, y_pred: list) -> float:
    cos_sim = list_of_cosine_similarities(y_true, y_pred)

    return statistics.median(cos_sim)


def minimum_cosine_similarity(y_true: list, y_pred: list) -> float:
    cos_sim = list_of_cosine_similarities(y_true, y_pred)

    return min(cos_sim)


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


def size(num_clusters: int, num_outliers: int, datatype: str, num_records: int, y_label: str, predicting_feature_count=1) -> tuple[float, float, float, float, float, int, int]:
    ml_models = {
        "lstm": 220_000,  # Size to store lstm model
        "linear_regression": 16,  # Size to store linear_regression model
        "multivariable LR": (8 * predicting_feature_count) + 8 # (slope * predictor) + intercept
    }
    datatype_length = size = np.dtype(datatype).itemsize

    cost_per_record_in_bits = math.ceil(
        math.log2(num_clusters + 1))  # cost associated with storing clustering information
    total_bytes_for_clusters = (cost_per_record_in_bits * num_records) // 8

    model_function = config_get("machine_learning_model")

    # Check if one global model type is used or if each column is specified
    use_one_model_type = config_get("use_one_model_type")
    if not use_one_model_type:
        predictors_dict = config_get("predicted_by")
        model_function = predictors_dict[y_label]['model']

    total_for_storing_models = num_clusters * ml_models[model_function]
    total_for_outliers = num_outliers * datatype_length

    original_size = datatype_length * num_records
    total_size = total_bytes_for_clusters + total_for_storing_models + total_for_outliers
    size_as_percentage = round((total_size / original_size) * 100, 4)

    return total_size, original_size, size_as_percentage, total_for_storing_models, total_for_outliers, datatype_length * 8, cost_per_record_in_bits


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


def size_text(num_clusters: int, outlier_records: list, original_records: pd.DataFrame, word2vec_model_name: str) -> tuple[float, float, float]:
    ml_models = {
        "lstm": 220_000,  # Size to store lstm model
        "linear_regression": 16,  # Size to store linear_regression model
    }

    num_records = len(original_records)

    cost_per_record_in_bits = math.ceil(
        math.log2(num_clusters + 1))  # cost associated with storing clustering information
    total_bytes_for_clusters = (cost_per_record_in_bits * num_records) // 8

    total_for_storing_models = num_clusters * ml_models[config_get('machine_learning_model')]
    total_for_outliers = sum(len(s) for s in outlier_records)

    total_for_word2vec_models = os.stat(word2vec_model_name).st_size if word2vec_model_name else 0

    original_size = sum(len(s) for s in original_records)
    total_size = total_bytes_for_clusters + total_for_storing_models + total_for_outliers + total_for_word2vec_models
    size_as_percentage = round((total_size / original_size) * 100, 4)

    return total_size, original_size, size_as_percentage


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
