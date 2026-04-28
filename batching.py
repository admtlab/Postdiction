import numpy as np
import pandas as pd
from intervaltree import IntervalTree
import time
from math import ceil
from joblib import Parallel, delayed
import tqdm
import sys
import statistics

from numpy.core.numeric import NaN
from sklearn.neighbors import KDTree
from tensorflow.keras.preprocessing.sequence import pad_sequences

import models
from helper import list_of_percent_differences
from models import Cluster
from multiprocessing import Pool, Lock, Value
from config import config_get
from text import cosine_similarity

clusters_lock = Lock()
file_case_study = Lock()
case_study_counter = Value('i', 0)
column_index_variable = config_get('index_column_name')
use_exhaustive_fallback = config_get('exhaustive_check_fallback')


def row_in_cluster(x_value, y_value, index, batch_method: str, sorted_clusters: list[models.Cluster], x_tree: np.array,
                   y_tree: list, kd_tree,
                   error_tolerance: float, model_type: str, null_method: str, replacement_x, replacement_y,
                   provide_error=False):
    is_outlier = False
    cluster_id = 0  # 0 is a special value for outliers. All other indexes will be +1
    prediction_error = 0
    interval_check_start = time.time()

    # Edge case for null values
    if null_method == 'outliers':
        if hasattr(x_value, '__len__') or hasattr(y_value, '__len__'):
            if pd.isnull(x_value.any()) or pd.isnull(y_value.any()):
                # Remove nulls as outliers
                interval_check_end = time.time()
                return interval_check_end - interval_check_start, index, True, cluster_id, prediction_error
        elif pd.isnull(x_value) or pd.isnull(y_value):
            # Remove nulls as outliers
            interval_check_end = time.time()
            return interval_check_end - interval_check_start, index, True, cluster_id, prediction_error
    elif null_method == 'replace_missing_value':
        if hasattr(x_value, '__len__') or hasattr(y_value, '__len__'):
            if pd.isnull(x_value.any()) or pd.isnull(y_value.any()):
                # Replace null values with a specified value
                x_value = replacement_x
                y_value = replacement_y
        elif pd.isnull(x_value) or pd.isnull(y_value):
            # Replace null values with a specified value
            x_value = replacement_x
            y_value = replacement_y
    if batch_method == "binary_search":
        found_value, cluster_index = value_exists_in_cluster(sorted_clusters, x_value, y_value, error_tolerance, index)
        if not found_value:
            is_outlier = True
        else:
            cluster_id = cluster_index
    elif batch_method == "tree_index":
        found_value, cluster_index = interval_index_check(x_value, y_value, x_tree)
        if not found_value:
            is_outlier = True
        else:
            cluster_id = cluster_index
    elif batch_method == "array_index":
        prechecked_clusters = []
        found_value, cluster_index, predicted_error, prechecked_clusters = interval_index_check_array(x_value, y_value,
                                                                                                      error_tolerance,
                                                                                                      index, x_tree,
                                                                                                      y_tree,
                                                                                                      provide_error=provide_error,
                                                                                                      model_type=model_type)

        if use_exhaustive_fallback and not found_value:
            # Remove previously checked clusters
            sorted_clusters = list(filter(lambda tmp_cluster: tmp_cluster.cluster_index not in prechecked_clusters,
                                          sorted_clusters))

            for cluster in sorted_clusters:
                found_value, _, predicted_error = check_model_within_threshold(cluster, x_value, y_value,
                                                                               error_tolerance,
                                                                               model_type=model_type,
                                                                               provide_error=provide_error)
                if found_value:
                    cluster_id = cluster.cluster_index
                    break

        if not found_value:
            is_outlier = True
        else:
            cluster_id = cluster_index
            prediction_error = predicted_error
    elif batch_method == "kd_tree_index":
        try:
            if not isinstance(x_value, list):
                x_value = x_value.tolist()
        except AttributeError:
            pass
        try:
            if not isinstance(y_value, list):
                y_value = y_value.reshape(-1).tolist()
        except AttributeError:
            pass

        if not isinstance(x_value, list) and not isinstance(y_value, list):
            values_to_check = np.array([x_value] + [y_value]).reshape(1, -1)
        else:
            values_to_check = np.array(x_value + y_value).reshape(1, -1)
        # Find nearest half of clusters to check, sort by least Minkowski distance
        is_in_cluster = False

        # Maintain cluster indexes checked by kd-tree. Small optimization if having to utilize exhaustive search
        prechecked_clusters = []
        if len(sorted_clusters) > 1:
            dist, ind = kd_tree.query(values_to_check, k=(len(sorted_clusters) // 2))
            for cluster_index in ind.tolist():
                cluster_to_check = sorted_clusters[cluster_index[0]]
                is_in_cluster, predicted_val, predicted_error = check_model_within_threshold(cluster_to_check, x_value, y_value,
                                                                                 error_tolerance,
                                                                                 model_type=model_type,
                                                                                 provide_error=provide_error)

                if is_in_cluster:
                    cluster_id = cluster_to_check.cluster_index
                    break
                prechecked_clusters.append(cluster_index)

        # If kd clusters do not match, do exhaustive check. Prioritize storage over speed
        if (use_exhaustive_fallback and not is_in_cluster) or (len(sorted_clusters) <= 1):
            is_in_cluster = False
            # Remove previously checked clusters
            sorted_clusters = list(filter(lambda tmp_cluster: tmp_cluster.cluster_index not in prechecked_clusters,
                                          sorted_clusters))
            for cluster in sorted_clusters:
                is_in_cluster, predicted_val, predicted_error = check_model_within_threshold(cluster, x_value, y_value,
                                                                                 error_tolerance,
                                                                                 model_type=model_type,
                                                                                 provide_error=provide_error)
                if is_in_cluster:
                    cluster_id = cluster.cluster_index
                    # # Decode and print the first 10 rows to reach this point
                    # with case_study_counter.get_lock():
                    #     current_count = case_study_counter.value
                    #     if current_count < 10:
                    #         case_study_counter.value += 1
                    #         should_log = True
                    #     else:
                    #         should_log = False
                    #
                    # if should_log:
                    #     val_to_display = []
                    #     for predicted_int in predicted_val:
                    #         predicted_sequence = int(predicted_int).to_bytes(
                    #             length=((int(predicted_int).bit_length() + 7) // 8),
                    #             byteorder='big', signed=False
                    #         )
                    #         predicted_dec = predicted_sequence.decode(encoding='utf-8', errors='backslashreplace')
                    #         val_to_display.append(predicted_dec)
                    #     file_case_study.acquire()
                    #     try:
                    #         with open("case_study.txt", "a") as case_file:
                    #             case_file.write(f"Decoded string for index {index} is: {val_to_display}\n")
                    #     finally:
                    #         file_case_study.release()
                    break

        if not is_in_cluster:
            is_outlier = True
    elif batch_method == "exhaustive_search":
        is_in_cluster = False
        for cluster in sorted_clusters:
            is_in_cluster, _, predicted_error = check_model_within_threshold(cluster, x_value, y_value, error_tolerance,
                                                                             model_type=model_type,
                                                                             provide_error=provide_error)
            if is_in_cluster:
                cluster_id = cluster.cluster_index
                prediction_error = predicted_error
                break

        if not is_in_cluster:
            is_outlier = True
    else:
        print("No such batch_method exists")

    interval_check_end = time.time()
    return interval_check_end - interval_check_start, index, is_outlier, cluster_id, prediction_error


def multivariable_row_in_cluster(x_values, y_value, index, batch_method: str, sorted_clusters: list[models.Cluster],
                                 x_tree: np.array,
                                 y_tree: list, kd_tree,
                                 error_tolerance: float, model_type: str, null_method: str, replacement_x: list,
                                 replacement_y):
    assert model_type == 'multivariable LR'

    is_outlier = False
    cluster_id = 0  # 0 is a special value for outliers
    interval_check_start = time.time()

    # Edge case for null values
    if null_method == 'outliers':
        # and pd.isnull(x_value) or pd.isnull(y_value):
        if pd.isnull(y_value):
            # Remove nulls as outliers
            interval_check_end = time.time()
            return interval_check_end - interval_check_start, index, True

        for cur_x_val in x_values:
            if pd.isnull(cur_x_val):
                # Remove nulls as outliers
                interval_check_end = time.time()
                return interval_check_end - interval_check_start, index, True

        # If here, no missing values to treat this row immediately as an outlier
    elif null_method == 'replace_missing_value':
        # and pd.isnull(x_value) or pd.isnull(y_value):
        if pd.isnull(y_value):
            y_value = replacement_y

        for cur_x_val, j in enumerate(x_values):
            if pd.isnull(cur_x_val):
                x_values[j] = replacement_x[j]

    if batch_method == "binary_search":
        raise ValueError('Not currently supported in multivariable setup')
    elif batch_method == "tree_index":
        raise ValueError('Not currently supported in multivariable setup')
    elif batch_method == "array_index":
        raise ValueError('Not currently supported in multivariable setup')
    elif batch_method == "exhaustive_search":
        is_in_cluster = False
        for cluster in sorted_clusters:
            is_in_cluster, _ = multivariable_check_model_within_threshold(cluster, x_values, y_value, error_tolerance,
                                                                          model_type=model_type)
            if is_in_cluster:
                cluster_id = cluster.cluster_index
                break

        if not is_in_cluster:
            is_outlier = True
    elif batch_method == "kd_tree_index":
        values_to_check = np.array(x_values + [y_value]).reshape(1, -1)
        # Find nearest half of clusters to check, sort by least Minkowski distance
        is_in_cluster = False

        # Maintain cluster indexes checked by kd-tree. Small optimization if having to utilize exhaustive search
        # raise ValueError(f'batched: {len(values_to_check[0])}')
        prechecked_clusters = []
        if len(sorted_clusters) > 1:
            dist, ind = kd_tree.query(values_to_check, k=(len(sorted_clusters) // 2))
            for cluster_index in ind.tolist():
                cluster_to_check = sorted_clusters[cluster_index[0]]
                is_in_cluster, _ = multivariable_check_model_within_threshold(cluster_to_check, x_values, y_value,
                                                                              error_tolerance, model_type=model_type)

                if is_in_cluster:
                    cluster_id = cluster_to_check.cluster_index
                    break
                prechecked_clusters.append(cluster_index)

        # If kd clusters do not match, do exhaustive check. Prioritize storage over speed
        if (use_exhaustive_fallback and not is_in_cluster) or (len(sorted_clusters) <= 1):
            is_in_cluster = False
            # Remove previously checked clusters
            sorted_clusters = list(
                filter(lambda tmp_cluster: tmp_cluster.cluster_index not in prechecked_clusters, sorted_clusters))
            for cluster in sorted_clusters:
                is_in_cluster, _ = multivariable_check_model_within_threshold(cluster, x_values, y_value,
                                                                              error_tolerance, model_type=model_type)
                if is_in_cluster:
                    cluster_id = cluster.cluster_index
                    break

        if not is_in_cluster:
            is_outlier = True
    else:
        print("No such batch_method exists")

    interval_check_end = time.time()
    return interval_check_end - interval_check_start, index, is_outlier, cluster_id


def rows_in_cluster(x_values, y_values, sorted_clusters: list[models.Cluster], error_tolerance: float):
    return check_models_within_threshold(sorted_clusters, x_values, y_values, error_tolerance)


def unpack_batch_args(args):
    return row_in_cluster(*args)


def unpack_multivariable_batch_args(args):
    return multivariable_row_in_cluster(*args)


def fit_new_batch(new_data: pd.DataFrame, batch_method: str, clusters: list[models.Cluster],
                  x_label: str, y_label: str, x_tree: np.array, y_tree: list, kdtree, error_tolerance: float,
                  verbose=False, clustering_destination=None, cur_partition_num=0, model_type='linear_regression',
                  null_method='outliers', replacement_x=None, replacement_y=None, provide_error=False) -> tuple[
    list, list[tuple], list[float], list[float]]:
    """
    Fit new data points into existing clusters or mark them as outliers. Batch method determines which method is used for fitting the new data.

    Returns:
        A tuple with the first element being the updated list of outliers and the second element being the updated
        list of cluster_indexes.
    """
    # Start timing fit_new_batch
    fit_batch_start = time.time()

    # Initialize the variables
    clusters = clusters
    if batch_method != 'kd_tree_index':
        sorted_clusters = sorted(clusters, key=lambda obj: obj.sample_y_value.all())
    else:
        sorted_clusters = clusters

    # Clear clusters for batch fitting
    if cur_partition_num == 0 and batch_method != "binary_search":
        for cluster in sorted_clusters:
            # Does not persist values since this is before each partition is processed, from sampling clusters
            cluster.flush_cluster(clustering_destination, y_label, cur_partition_num)

    # Loop through new data points
    batch_rows = []
    if model_type != 'lstm':
        for row in tqdm.tqdm(new_data.itertuples(), desc='Preparing Rows for Processing', colour='red', leave=False):
            x_value = getattr(row, x_label)
            y_value = getattr(row, y_label)
            index = getattr(row, column_index_variable)
            batch_rows.append(
                (x_value, y_value, index, batch_method, sorted_clusters, x_tree, y_tree, kdtree, error_tolerance,
                 model_type,
                 null_method, replacement_x, replacement_y, provide_error))

    process_results = []
    if model_type != 'lstm':
        n_procs = max(1, config_get('num_processes'))
        if len(batch_rows) > 0:
            chunksize = max(1, ceil(len(batch_rows) / (n_procs * 4)))
            with Pool(n_procs) as p:
                for result in tqdm.tqdm(p.imap_unordered(unpack_batch_args, batch_rows, chunksize=chunksize),
                                        total=len(batch_rows), desc='Processing Rows', colour='green', leave=False):
                    process_results.append(result)

    else:
        process_results = rows_in_cluster(new_data[x_label], new_data[y_label], sorted_clusters, error_tolerance)

    row_times = list(map(lambda x: x[0], process_results))
    row_outliers = list(filter(lambda x: x[2] == True, process_results))
    row_inliers = list(filter(lambda x: x[2] == False, process_results))
    batch_outliers = (list(map(lambda x: x[1], row_outliers)))

    outlier_errors = [0]
    inlier_errors = [0]

    if provide_error:
        outlier_errors = [x[4] for x in row_outliers]
        inlier_errors = [x[4] for x in row_inliers]

    # (Row Index, Cluster Index)
    batch_inlier_cluster_indexes = (list(map(lambda x: (x[1], x[3]), row_inliers)))

    # End timing fit_new_batch
    fit_batch_end = time.time()

    # Calculate the total time spent in fit_new_batch
    total_fit_batch_time = fit_batch_end - fit_batch_start

    if verbose:
        avg_row_time = sum(row_times) / len(row_times)
        print(f"Total time taken in fit_new_batch: {total_fit_batch_time:.4f} seconds")
        print(f"Time spent in interval_index_check: {avg_row_time:.4f} seconds")
        print(f"Proportion of time in interval_index_check: {(avg_row_time / total_fit_batch_time):.4%}")

    # Clear clusters for new batch fitting
    if batch_method == "binary_search":
        for cluster in sorted_clusters:
            cluster.flush_cluster(clustering_destination, y_label, cur_partition_num)

    return batch_outliers, batch_inlier_cluster_indexes, outlier_errors, inlier_errors


def multivariable_fit_new_batch(new_data: pd.DataFrame, batch_method: str, clusters: list[models.Cluster],
                                x_label: list[str], y_label: str, x_tree: np.array, y_tree: list, kd_tree: KDTree,
                                error_tolerance: float,
                                verbose=False, clustering_destination=None, cur_partition_num=0,
                                model_type='linear_regression',
                                null_method='outliers', replacement_x: list = None, replacement_y=None) -> tuple[
    list, list[tuple]]:
    """
    Fit new data points into existing clusters or mark them as outliers. Batch method determines which method is used for fitting the new data.

    Returns:
        A tuple with the first element being the updated list of clusters and the second element being the updated
        list of outliers.
    """
    assert model_type == 'multivariable LR'

    # Start timing fit_new_batch
    fit_batch_start = time.time()

    # Initialize the variables
    clusters = clusters
    sorted_clusters = sorted(clusters, key=lambda obj: obj.sample_y_value)

    # Clear clusters for batch fitting
    if cur_partition_num == 0 and batch_method != "binary_search":
        for cluster in sorted_clusters:
            # Does not persist values since this is before each partition is processed, from sampling clusters
            cluster.flush_cluster(clustering_destination, y_label, cur_partition_num)

    # Loop through new data points
    batch_rows = []
    for row in tqdm.tqdm(new_data.itertuples(), desc='Preparing Rows for Processing', colour='red', leave=False):
        x_val_lst = [getattr(row, x_lab) for x_lab in x_label]
        y_value = getattr(row, y_label)
        index = getattr(row, column_index_variable)
        # print(x_val_lst)
        batch_rows.append(
            (x_val_lst, y_value, index, batch_method, sorted_clusters, x_tree, y_tree, kd_tree, error_tolerance,
             model_type,
             null_method, replacement_x, replacement_y))

    process_results = []

    n_procs = max(1, config_get('num_processes'))
    if len(batch_rows) > 0:
        chunksize = max(1, ceil(len(batch_rows) / (n_procs * 4)))
        with Pool(n_procs) as p:
            for result in tqdm.tqdm(p.imap_unordered(unpack_multivariable_batch_args, batch_rows, chunksize=chunksize),
                                    total=len(batch_rows), desc='Processing Rows', colour='green', leave=False):
                process_results.append(result)

    row_times = list(map(lambda x: x[0], process_results))
    row_outliers = list(filter(lambda x: x[2] == True, process_results))
    batch_outliers = (list(map(lambda x: x[1], row_outliers)))
    row_inliers = list(filter(lambda x: x[2] == False, process_results))

    # (Row Index, Cluster Index)
    batch_inlier_cluster_indexes = (list(map(lambda x: (x[1], x[3]), row_inliers)))

    # End timing fit_new_batch
    fit_batch_end = time.time()

    # Calculate the total time spent in fit_new_batch
    total_fit_batch_time = fit_batch_end - fit_batch_start

    if verbose:
        avg_row_time = sum(row_times) / len(row_times)
        print(f"Total time taken in fit_new_batch: {total_fit_batch_time:.4f} seconds")
        print(f"Time spent in interval_index_check: {avg_row_time:.4f} seconds")
        print(f"Proportion of time in interval_index_check: {(avg_row_time / total_fit_batch_time):.4%}")

    # Clear clusters for new batch fitting
    if batch_method == "binary_search":
        for cluster in sorted_clusters:
            cluster.flush_cluster(clustering_destination, y_label, cur_partition_num)

    return batch_outliers, batch_inlier_cluster_indexes


def create_interval_index(batch_method: str, data: pd.DataFrame, clusters: list[models.Cluster], x_label: str,
                          error_tolerance: float, density_split=1000, leaf_level=40) -> tuple[np.array, list, KDTree]:
    """
    Creates the interval-based index (based on the x-axis and y-axis) depending on the provided batch_method.

    Returns:
        A tuple where the first element is the indexes along the x-axis as a Numpy Array and the second element is
        the indexes along the y-axis as a list. If KD-Tree is used, the third argument is the KD Tree. If the indexes
        are not applicable to the batch_method, the resulting tuple element is set to None.
    """
    x_index = None
    y_index = None
    kd_index = None
    if batch_method == "tree_index":
        x_index = create_interval_tree(data, clusters, x_label, error_tolerance)
    elif batch_method == "array_index":
        x_index, y_index = create_range_array_index(data, clusters, x_label, error_tolerance)
    elif batch_method == "binary_search":
        pass
    elif batch_method == "kd_tree_index":
        kd_index = create_kd_tree_index(clusters, leaf_level)

    return x_index, y_index, kd_index


"""
================================
Batch Method: K-D Tree Index
================================
"""


def create_kd_tree_index(clusters: np.array, leaf_level=40):
    # For each cluster, obtain the representative point
    cluster_representative_points = [cur_cluster.representative_point for cur_cluster in clusters]
    tree = KDTree(cluster_representative_points, leaf_level)
    return tree


"""
================================
Batch Method: Array Index
================================
"""


def create_range_array_index(data: pd.DataFrame, clusters: np.array, x_label: str, error_tolerance: float,
                             density_split=1000) -> tuple[np.array, list]:
    """
    Creates an indexed range array based on the provided x-values and clusters.
    Optimized for large cluster counts using direct coefficient access.
    """
    # Sort the data based on x_label
    is_x_a_vector = hasattr(data.loc[0, x_label], '__len__')

    if is_x_a_vector:
        sorted_data = data.sort_values(by=x_label, key=lambda s: s.apply(tuple))
    else:
        sorted_data = data.sort_values(by=x_label)

    # Calculate the number of points per interval
    total_points = len(sorted_data)
    points_per_interval = max(1, total_points // density_split)

    x_min_values = []
    y_max_min_values = []
    prev_x_max = None

    # Pre-extract all model coefficients (slope, intercept) once
    slopes = np.array([cluster.model.coef_[0] for cluster in clusters])
    intercepts = np.array([cluster.model.intercept_ for cluster in clusters])

    # Loop through the sorted_data in chunks
    i = 0
    while i < total_points:
        chunk = sorted_data.iloc[i:i + points_per_interval]
        if is_x_a_vector:
            chunk_as_tuples = chunk[x_label].apply(tuple)
            x_min, x_max = chunk_as_tuples.min(), chunk_as_tuples.max()
        else:
            x_min, x_max = chunk[x_label].min(), chunk[x_label].max()

        # If the x_min and x_max are the same, adjust the range
        while x_min == x_max and i + points_per_interval < total_points:
            i += points_per_interval
            next_chunk = sorted_data.iloc[i:i + points_per_interval]
            if is_x_a_vector:
                next_chunk_as_tuples = next_chunk[x_label].apply(tuple)
                x_max = next_chunk_as_tuples.max()
            else:
                x_max = next_chunk[x_label].max()

        # Edge case where the last interval is still zero-width
        if x_min == x_max:
            x_max += 0.1

        # Ensure continuity
        if prev_x_max is not None and x_min < prev_x_max:
            x_min = prev_x_max

        x_min_values.append(x_min)
        prev_x_max = x_max

        y_predictions_min = []
        y_predictions_max = []
        predicting_feature_count = 1

        if is_x_a_vector:
            for cluster in clusters:
                y_value_at_x_min = cluster.model.predict([x_min])[0]
                y_value_at_x_max = cluster.model.predict([x_max])[0]
                y_predictions_min.append(y_value_at_x_min)
                y_predictions_max.append(y_value_at_x_max)
                predicting_feature_count = cluster.predicting_feature_count
        else:
            # Direct algebraic calculation instead of predict()
            y_predictions_min = slopes * x_min + intercepts
            y_predictions_max = slopes * x_max + intercepts

        # Vectorized slope and bounds calculation
        if is_x_a_vector:
            tuple_slope = tuple(0 for _ in range(0, predicting_feature_count))
            is_positive_slope = slopes > tuple_slope
        else:
            is_positive_slope = slopes > 0

        y_min_list = np.zeros(len(clusters))
        y_max_list = np.zeros(len(clusters))

        # Apply logic based on slope direction (vectorized)
        y_min_list[is_positive_slope] = y_predictions_max[is_positive_slope] * (1 - error_tolerance)
        y_max_list[is_positive_slope] = y_predictions_min[is_positive_slope] * (1 + error_tolerance)

        y_min_list[~is_positive_slope] = y_predictions_min[~is_positive_slope] * (1 - error_tolerance)
        y_max_list[~is_positive_slope] = y_predictions_max[~is_positive_slope] * (1 + error_tolerance)

        # Filter valid clusters (where y_min < y_max)
        valid_mask = y_min_list < y_max_list
        valid_indices = np.where(valid_mask)[0]

        cluster_list = [clusters[idx] for idx in valid_indices]

        y_max_min_values.append((
            y_min_list[valid_indices],
            y_max_list[valid_indices],
            cluster_list
        ))

        # Move to the next chunk
        i += points_per_interval

    return np.array(x_min_values), y_max_min_values


def interval_index_check_array(x_value: float, y_value: float, error_tolerance: float, value_index: int,
                               x_min_array: np.array,
                               y_max_min_values: list, verbose=False, provide_error=False,
                               model_type='linear_regression') -> tuple[bool, int, float | None, list]:
    """
    Checks if the provided x and y values fall within the specified range intervals.
    Optimized with binary search and vectorized operations.
    """
    # Use binary search instead of linear search
    index = np.searchsorted(x_min_array, x_value)

    # Adjust index if we need the interval containing x_value
    if index > 0 and index < len(x_min_array):
        # Check if x_value is in the interval [x_min_array[index-1], x_min_array[index]]
        index = index - 1
    elif index >= len(x_min_array):
        index = len(x_min_array) - 1

    # Edge Case: If the value is not found or there are no existing y-clusters at the specified x-range
    if index < 0 or index >= len(y_max_min_values):
        return False, 0, 0, []

    y_info = y_max_min_values[index]
    y_min_array = y_info[0]
    y_max_array = y_info[1]
    cluster_list = y_info[2]

    # Vectorized boolean check instead of looping
    # Find all clusters where y_value falls within bounds
    is_within = (y_value >= y_min_array) & (y_value <= y_max_array)

    if not np.any(is_within):
        if verbose:
            print("not found in y arrays")
        return False, 0, 0, []

    # Get the first matching cluster index
    # cluster_index = np.argmax(is_within)  # Returns first True index
    # cluster = cluster_list[cluster_index]

    lower_bound = y_value * (1 - (error_tolerance * 1e-2))
    upper_bound = y_value * (1 + (error_tolerance * 1e-2))

    candidate_indices = np.where(is_within)[0]
    for ci in candidate_indices:
        cluster = cluster_list[ci]

        if model_type == 'linear_regression':
            predicted_y = cluster.model.coef_[0] * x_value + cluster.model.intercept_
        else:
            predicted_y = cluster.model.predict([[x_value]])[0]

        if lower_bound <= predicted_y <= upper_bound:
            prediction_error = 0
            if provide_error:
                prediction_error = list_of_percent_differences([y_value], [predicted_y])[0]

            # Lock and update cluster size
            clusters_lock.acquire()
            try:
                cluster.size += 1
            finally:
                clusters_lock.release()

            return True, cluster.cluster_index, prediction_error, candidate_indices.tolist()

    # In the future, you will need to use add_new_value() function so that it can
    # track the added index like this. Make sure to remove the line
    # cluster.size += 1 as this logic is achieved by the function below
    # cluster.add_new_value(value_index)

    return False, 0, 0, candidate_indices.tolist()


def find_y_index(y_min_array: np.array, y_max_array: np.array, y_value: float) -> int:
    """
    Finds the index of a y-value within the y_min and y_max arrays.

    Args:
        y_min_array (np.array): Array of minimum y-values for each interval.
        y_max_array (np.array): Array of maximum y-values for each interval.
        y_value (float): The y-value to check.

    Returns:
        int: The index where the y_value falls within the y_min and y_max bounds, or -1 if not found.
    """
    indices = np.where((y_value >= y_min_array) & (y_value <= y_max_array))[0]
    return indices[0] if len(indices > 0) else -1


def find_x_index(arr: np.array, x: float) -> int:
    """
    Finds the index of an x-value within an array of x_min values.

    Args:
        arr (np.array): Array of x_min values.
        x (float): The x-value to check.

    Returns:
        int: The index where the x_value falls between two consecutive values in the array, or -1 if not found.
    """
    if len(arr) == 1 and arr[0] == 0 and x == 0:
        return 0

    for i in range(len(arr) - 1):
        if arr[i] <= x <= arr[i + 1]:
            return i
    return -1  # Return -1 if no valid index is found


"""
================================
Batch Method: Binary Search
================================
"""


def value_exists_in_cluster(sorted_clusters: list, x_value: float, y_value: float, error_tolerance: float,
                            index: int) -> tuple[bool, int]:
    """
    Check if the (x_value, y_value) exists in a cluster within the error tolerance and add the value if found.

    Args:
        sorted_clusters (list): List of clusters sorted by sample_y_value.
        x_value (float): The x-axis value to be matched.
        y_value (float): The y-axis value to be matched.
        error_tolerance (float): The acceptable margin of error for fitting the value into a cluster.
        index (int): The index of the new data point being evaluated.

    Returns:
        bool: True if the value exists in a cluster within the error tolerance and is added; False otherwise.
    """

    cluster, predicted_y_val, cl_index = find_exact_cluster(sorted_clusters, x_value, y_value, error_tolerance)

    if cluster is not None:
        cluster.add_new_value(index, y_value, predicted_y_val)
        return True, cluster.cluster_index

    return False, 0


def find_exact_cluster(sorted_clusters: list, x_value: float, y_value: float, error_tolerance: float) -> tuple[
                                                                                                             models.Cluster, float, int | None] | \
                                                                                                         tuple[
                                                                                                             None, float, int | None]:
    """
    Find the exact cluster that best matches the given (x_value, y_value) within the error tolerance.

    Args:
        sorted_clusters (list): List of clusters sorted by sample_y_value.
        x_value (float): The x-axis value to be matched.
        y_value (float): The y-axis value to be matched.
        error_tolerance (float): The acceptable margin of error for fitting the value into a cluster.

    Returns:
        tuple: A tuple containing the matching cluster (or None if no match) and the predicted y_value.
    """

    index = binary_search_of_clusters(sorted_clusters, y_value, error_tolerance)

    if index is None:
        return None, 0, index  # Return None instead of False for consistency

    # Calculate tolerance range for the y_value
    lower_bound = y_value * (1 - error_tolerance)
    upper_bound = y_value * (1 + error_tolerance)

    # Step 2: Start checking from the found index and around it
    n = len(sorted_clusters)

    # Check the cluster at the found index
    is_within_threshold, predicted_y, _ = check_model_within_threshold(sorted_clusters[index], x_value, y_value,
                                                                       error_tolerance)
    if is_within_threshold:
        return sorted_clusters[index], predicted_y, index

    # Step 3: If not within bounds, explore surrounding clusters
    # Check previous clusters (leftwards in the sorted list)
    for i in range(index - 1, -1, -1):
        is_within_threshold, predicted_y, _ = check_model_within_threshold(sorted_clusters[i], x_value, y_value,
                                                                           error_tolerance)
        if predicted_y > upper_bound:
            # We've gone too far left, no need to check further
            break
        if is_within_threshold:
            return sorted_clusters[i], predicted_y, index

    # Check next clusters (rightwards in the sorted list)
    for i in range(index + 1, n):
        is_within_threshold, predicted_y, _ = check_model_within_threshold(sorted_clusters[i], x_value, y_value,
                                                                           error_tolerance)
        if predicted_y < lower_bound:
            # We've gone too far right, no need to check further
            break
        if is_within_threshold:
            return sorted_clusters[i], predicted_y, index

    # Step 4: If no suitable cluster is found, return None and a default predicted value
    return None, 0, index


def binary_search_of_clusters(sorted_clusters: list, y_value: float, error_tolerance: float) -> int | None:
    """
    Perform a binary search to find the cluster with a sample_y_value close to the target y_value within the error tolerance.

    Args:
        sorted_clusters (list): List of clusters sorted by sample_y_value.
        y_value (float): The target y_value to search for.
        error_tolerance (float): The acceptable margin of error for finding a matching cluster.

    Returns:
        int or None: The index of the cluster that falls within the error tolerance, or None if no match is found.
    """

    low, high = 0, len(sorted_clusters) - 1

    # Calculate tolerance range
    lower_bound = y_value * (1 - error_tolerance)
    upper_bound = y_value * (1 + error_tolerance)

    # Perform binary search
    while low <= high:
        mid = (low + high) // 2
        cluster_y_value = sorted_clusters[mid].sample_y_value  # Assuming y_values is a list

        if lower_bound <= cluster_y_value <= upper_bound:
            # Found a cluster within the tolerance range
            return mid
        elif cluster_y_value < lower_bound:
            # Move to the right half
            low = mid + 1
        else:
            # Move to the left half
            high = mid - 1

    # Return None. Don't return mid since it would be difficult to determine if a cluster was not found vs. false positive
    return None


def check_model_within_threshold(cluster: models.Cluster, x_value: float, y_value: float, error_tolerance: float,
                                 model_type=None, provide_error=False) -> \
        tuple[bool, float, float]:
    """
    Check if the predicted y_value from the cluster's model is within the error tolerance.

    Args:
        cluster (Cluster): The cluster containing the model used to predict the y_value.
        x_value (float): The x_value used for prediction.
        y_value (float): The target y_value for comparison.
        error_tolerance (float): The acceptable margin of error for matching the predicted y_value.

    Returns:
        - bool: True if the predicted y_value is within the error tolerance, False otherwise.
        - float: The predicted y_value from the model.
    """
    predicted_y: float
    if model_type == 'lstm':
        original_x_value = np.array(x_value).reshape(-1, 1)

        X = np.array(original_x_value)
        predicted_y = cluster.model.predict(X)[0]
        lower_bound = y_value * (1 - (error_tolerance * 1e-2))
        upper_bound = y_value * (1 + (error_tolerance * 1e-2))

        prediction_error = 0
        if provide_error:
            prediction_error = list_of_percent_differences([y_value], [predicted_y])[0]

        # Return both the check result and the predicted_y
        return lower_bound <= predicted_y <= upper_bound, predicted_y, prediction_error
    else:
        if hasattr(y_value, '__len__'):  # vector case
            predicted_y = cluster.model.predict([x_value])[0]

            # Convert to NumPy arrays
            y_vec = np.array(y_value)
            pred_vec = np.array(predicted_y)

            # Inline padding to match lengths
            if len(y_vec) != len(pred_vec):
                max_len = max(len(y_vec), len(pred_vec))
                if len(y_vec) < max_len:
                    y_vec = np.pad(y_vec, (0, max_len - len(y_vec)))
                else:
                    pred_vec = np.pad(pred_vec, (0, max_len - len(pred_vec)))

            # Cosine similarity
            dot = np.dot(y_vec, pred_vec)
            norms = np.linalg.norm(y_vec) * np.linalg.norm(pred_vec)

            # Handle edge cases, set cosine similarity to neutral since the values are technically undefined
            if norms == 0 or np.isinf(norms):
                cos_sim = 0.0
            else:
                cos_sim = dot / norms

            error_threshold = 1 - (error_tolerance * 1e-2)
            return cos_sim >= error_threshold, predicted_y, cos_sim

        else:
            predicted_y = cluster.model.predict([[x_value]])[0]
            lower_bound = y_value * (1 - (error_tolerance * 1e-2))
            upper_bound = y_value * (1 + (error_tolerance * 1e-2))

            prediction_error = 0
            if provide_error:
                prediction_error = list_of_percent_differences([y_value], [predicted_y])[0]

            # Return both the check result and the predicted_y
            return lower_bound <= predicted_y <= upper_bound, predicted_y, prediction_error


def multivariable_check_model_within_threshold(cluster: models.Cluster, x_values: list[float], y_value: float,
                                               error_tolerance: float,
                                               model_type=None) -> \
        tuple[bool, float]:
    """
    Check if the predicted y_value from the cluster's model is within the error tolerance.

    Args:
        cluster (Cluster): The cluster containing the model used to predict the y_value.
        x_value (float): The x_value used for prediction.
        y_value (float): The target y_value for comparison.
        error_tolerance (float): The acceptable margin of error for matching the predicted y_value.

    Returns:
        - bool: True if the predicted y_value is within the error tolerance, False otherwise.
        - float: The predicted y_value from the model.
    """
    assert model_type == 'multivariable LR'
    predicted_y: float = cluster.model.predict([x_values])[0]  # Predict y_value using the cluster's model
    lower_bound = y_value * (1 - error_tolerance)
    upper_bound = y_value * (1 + error_tolerance)

    # Return both the check result and the predicted_y
    return lower_bound <= predicted_y <= upper_bound, predicted_y


def check_models_within_threshold(clusters: list[models.Cluster], x_values: list, y_values: list,
                                  error_tolerance: float) -> list[tuple[float, int, bool, int]]:
    result_tuples: list[tuple[float, int, bool, int]] = []

    # Pre-fill result tuples
    for i in range(len(y_values)):
        result_tuples.append((0, i, True, -1))

    # Build X using the first cluster's model input shape
    model = clusters[0].model
    _, timesteps, features = model.input_shape

    # Convert x_values into a uniform NumPy structure
    # Case 1: scalar model
    if timesteps == 1 and features == 1:
        scalar_x = []
        for xv in x_values:
            if hasattr(xv, "__len__"):
                xv = np.array(xv, dtype=np.float64)
                xv = float(np.mean(xv))
            else:
                xv = float(xv)
            scalar_x.append(xv)

        X = np.array(scalar_x, dtype=np.float64).reshape(len(scalar_x), 1, 1)
    # Case 2: vector model
    elif timesteps == 1:
        vector_x = []
        for xv in x_values:
            xv = np.array(xv, dtype=np.float64)
            if xv.ndim == 1:
                vector_x.append(xv)
            else:
                vector_x.append(np.mean(xv, axis=0))
        vector_x = np.array(vector_x, dtype=np.float64)
        X = vector_x.reshape(len(vector_x), 1, features)
    # Case 3: sequence model
    else:
        X = pad_sequences(x_values, dtype='float64', padding='post')

        # Ensure padded length matches model timesteps
        if X.shape[1] != timesteps:
            if X.shape[1] > timesteps:
                X = X[:, :timesteps, :]
            else:
                pad_len = timesteps - X.shape[1]
                X = np.pad(X, ((0, 0), (0, pad_len), (0, 0)))

    # Build y-boundaries (CPU only)
    y_boundaries = []
    for y_val in y_values:
        if hasattr(y_val, "__len__"):
            y_boundaries.append(("vector", np.array(y_val)))
        else:
            lower = y_val * (1 - error_tolerance)
            upper = y_val * (1 + error_tolerance)
            y_boundaries.append(("scalar", (lower, upper)))

    # Evaluate each cluster without multiprocessing for TF
    for cluster in clusters:
        # Predict in the parent process (GPU-safe)
        interval_start = time.time()
        predicted_y_vals = cluster.model.predict(X)
        interval_end = time.time() - interval_start

        predicted_y_vals = predicted_y_vals.astype(np.float64)

        # Prepare CPU-only comparison tasks
        unzipped_list = [
            (i, predicted_y_vals[i], y_boundaries[i], error_tolerance)
            for i in range(len(predicted_y_vals))
        ]

        # Multiprocessing for CPU comparisons
        with Pool(config_get('num_processes')) as pool:
            for result in pool.imap(unpack_range_or_cosine, unzipped_list, chunksize=ceil(len(unzipped_list) / 22)):
                result_index = result[1]
                result_in_cluster = result[0]

                if result_tuples[result_index][2] and result_in_cluster:
                    result_tuples[result_index] = (interval_end, result_index, False, cluster.cluster_index)

        # Stop early if all matched
        if all(t[2] is False for t in result_tuples):
            break

    return result_tuples


def unpack_range_predictions(args):
    return predicted_is_in_range(*args)


def unpack_range_or_cosine(args):
    idx, predicted_y, y_boundary, error_tolerance = args
    mode, data = y_boundary

    if mode == "vector":
        y_vec = data
        pred_vec = np.array(predicted_y)

        # Pad to equal length
        if len(y_vec) != len(pred_vec):
            max_len = max(len(y_vec), len(pred_vec))
            y_vec = np.pad(y_vec, (0, max_len - len(y_vec)))
            pred_vec = np.pad(pred_vec, (0, max_len - len(pred_vec)))

        dot = np.dot(y_vec, pred_vec)
        norms = np.linalg.norm(y_vec) * np.linalg.norm(pred_vec)

        if norms == 0 or np.isinf(norms):
            cos_sim = 0.0
        else:
            cos_sim = dot / norms

        # Same threshold logic as your first block
        error_threshold = 1 - (error_tolerance * 1e-2)
        return (cos_sim >= error_threshold, idx)

    lower, upper = data
    return (lower <= predicted_y <= upper, idx)


def predicted_is_in_range(index: int, y_predicted: float, lower_bound: float, upper_bound: float):
    return lower_bound <= y_predicted <= upper_bound, index


"""
================================
Batch Method: Tree Index
================================
"""


def interval_index_check(x_value: float, y_value: float, interval_tree: IntervalTree, verbose=False) -> tuple[
    bool, int]:
    """
    Traverse the interval tree and add the value to a cluster if it falls within the correct x and y intervals.
    
    Args:
        x_value (float): The x-axis value of the point to be checked.
        y_value (float): The y-axis value of the point to be checked.
        interval_tree (IntervalTree): The interval tree where the x-intervals map to y-interval trees.
        verbose (bool): If True, print timing information; otherwise, do not print.
    
    Returns:
        bool: True if the point was added to a cluster, False if it was not found in any interval.
    """
    cluster_index = 0
    # Start total function timer
    total_start = time.perf_counter()

    # Step 1: Search for the x_value in the x-interval tree
    x_search_start = time.perf_counter()
    x_intervals = interval_tree.at(x_value)
    x_search_end = time.perf_counter()

    if not x_intervals:
        # If no x-interval was found, return False
        total_end = time.perf_counter()
        if verbose:
            print(f"Total time in interval_index_check: {(total_end - total_start) * 1e6:.2f} µs")
            print(f"Time spent in x-interval search: {(x_search_end - x_search_start) * 1e6:.2f} µs")
        return False, 0

    # Step 2: Loop through the found x-intervals
    for x_interval in x_intervals:
        # Get the corresponding y-interval tree for this x-interval
        y_tree_start = time.perf_counter()
        y_interval_tree = x_interval.data
        y_tree_end = time.perf_counter()

        # Step 3: Search for the y_value in the y-interval tree
        y_search_start = time.perf_counter()
        y_intervals = y_interval_tree.at(y_value)
        y_search_end = time.perf_counter()

        if y_intervals:
            # Step 4: If y_intervals are found, add the point to the corresponding cluster
            cluster_add_start = time.perf_counter()
            for y_interval in y_intervals:
                cluster = y_interval.data  # The cluster associated with this interval
                # Add the index (and y_value) to the cluster
                cluster.size += 1
                cluster_index = cluster.cluster_index
                break
            cluster_add_end = time.perf_counter()

            # Print timing for cluster addition and return True
            total_end = time.perf_counter()
            if verbose:
                print(f"Total time in interval_index_check: {(total_end - total_start) * 1e6:.2f} µs")
                print(f"Time spent in x-interval search: {(x_search_end - x_search_start) * 1e6:.2f} µs")
                print(f"Time spent getting y-interval tree: {(y_tree_end - y_tree_start) * 1e6:.2f} µs")
                print(f"Time spent in y-interval search: {(y_search_end - y_search_start) * 1e6:.2f} µs")
                print(f"Time spent adding to cluster: {(cluster_add_end - cluster_add_start) * 1e6:.2f} µs")
            return True, cluster_index

    # Step 5: If no matching y_interval was found, return False
    total_end = time.perf_counter()
    if verbose:
        print(f"Total time in interval_index_check: {(total_end - total_start) * 1e6:.2f} µs")
        print(f"Time spent in x-interval search: {(x_search_end - x_search_start) * 1e6:.2f} µs")

    return False, 0


def create_interval_tree(data: pd.DataFrame, clusters: list, x_label: str, error_tolerance: float,
                         density_split=1000) -> IntervalTree:
    """
    Create an interval tree that maps x-axis intervals to y-axis interval trees for finding the corresponding cluster for a point.
    
    Args:
        data (pd.DataFrame): The dataset to be partitioned.
        clusters (list): A list of pre-existing clusters, each with a model to predict y-values.
        x_label (str): The column name in `data` that represents the x-axis value of each data point.
        error_tolerance (float): The allowed margin of error for fitting data points into clusters.
        density_split (int): The number of intervals to create, each representing an equal number of data points.
        
    Returns:
        IntervalTree: The x-interval tree, where each x-interval contains a y-interval tree that maps y-ranges to clusters.
    """

    # Sort the data based on x_label
    sorted_data = data.sort_values(by=x_label)

    # Calculate the number of points per interval
    total_points = len(sorted_data)
    points_per_interval = total_points // density_split  # Divide data points evenly into intervals

    # Initialize the overall x-interval tree
    x_interval_tree = IntervalTree()

    # Track the previous x_max to ensure continuity between intervals
    prev_x_max = None

    # Loop through the sorted_data in chunks based on the calculated number of points per interval
    i = 0
    while i < total_points:
        chunk = sorted_data.iloc[i:i + points_per_interval]
        x_min, x_max = chunk[x_label].min(), chunk[x_label].max()

        # If the x_min and x_max are the same (i.e., the interval would collapse), we need to adjust
        while x_min == x_max and i + points_per_interval < total_points:
            # Expand the chunk by adding more points from the next batch
            i += points_per_interval
            next_chunk = sorted_data.iloc[i:i + points_per_interval]
            x_max = next_chunk[x_label].max()  # Expand x_max to create a non-zero interval

        # Edge case where the last interval is size 0
        if x_min == x_max:
            x_max += .1

        # Ensure continuity by connecting the current x_min with the previous x_max if necessary
        if prev_x_max is not None and x_min < prev_x_max:
            x_min = prev_x_max

        # Create a y-interval tree for this x-interval
        y_interval_tree = IntervalTree()

        # For each cluster, create a y-interval based on the x_min and x_max values
        for cluster in clusters:
            # Predict y-values for x_min and x_max using the cluster's model
            y_value_at_x_min = cluster.model.predict([[x_min]])[0]
            y_value_at_x_max = cluster.model.predict([[x_max]])[0]

            # Check the slope direction
            if y_value_at_x_max > y_value_at_x_min:  # Positive slope
                y_min = y_value_at_x_max * (1 - error_tolerance)
                y_max = y_value_at_x_min * (1 + error_tolerance)
            else:  # Negative slope
                y_min = y_value_at_x_min * (1 - error_tolerance)
                y_max = y_value_at_x_max * (1 + error_tolerance)

            # Add the y-interval for this cluster to the y_interval_tree
            if y_min < y_max:
                y_interval_tree.addi(y_min, y_max, cluster)
            else:
                pass

        # Add the x-interval to the x-interval tree, mapping to the corresponding y-interval tree
        x_interval_tree.addi(x_min, x_max, y_interval_tree)

        # Update prev_x_max to ensure continuity in the next iteration
        prev_x_max = x_max

        # Move to the next chunk
        i += points_per_interval

    return x_interval_tree
