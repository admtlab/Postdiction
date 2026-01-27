import numpy as np
import pandas as pd
from intervaltree import IntervalTree
import time

import models
from models import Cluster
from multiprocessing import Pool, Lock

clusters_lock = Lock()

def row_in_cluster(x_value, y_value, index, batch_method: str, sorted_clusters: list[models.Cluster], x_tree: np.array, y_tree: list,
                   error_tolerance: float):
    is_outlier = False
    interval_check_start = time.time()
    # Edge case for null values treated as Outliers
    if pd.isnull(x_value) or pd.isnull(y_value):
        is_outlier = True
    elif batch_method == "binary_search":
        if not value_exists_in_cluster(sorted_clusters, x_value, y_value, error_tolerance, index):
            is_outlier = True
    elif batch_method == "tree_index":
        if not interval_index_check(x_value, y_value, x_tree):
            is_outlier = True
    elif batch_method == "array_index":
        if not interval_index_check_array(x_value, y_value, index, x_tree, y_tree):
            is_outlier = True
    else:
        print("No such batch_method exists")

    interval_check_end = time.time()
    return interval_check_end - interval_check_start, index, is_outlier


def fit_new_batch(new_data: pd.DataFrame, batch_method: str, clusters: list[models.Cluster],
                  x_label: str, y_label: str, x_tree: np.array, y_tree: list, error_tolerance: float,
                  verbose=False, clustering_destination=None, cur_partition_num=0) -> list:
    """
    Fit new data points into existing clusters or mark them as outliers. Batch method determines which method is used for fitting the new data.

    Returns:
        A tuple with the first element being the updated list of clusters and the second element being the updated
        list of outliers.
    """
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
    for row in new_data.itertuples():
        x_value = getattr(row, x_label)
        y_value = getattr(row, y_label)
        index = getattr(row, "Index")
        batch_rows.append((x_value, y_value, index, batch_method, sorted_clusters, x_tree, y_tree, error_tolerance))

    process_results: list[tuple[float, list, bool]]
    with Pool(22) as p:
        process_results = p.starmap(row_in_cluster, batch_rows)

    row_times = list(map(lambda x: x[0], process_results))
    row_outliers = list(filter(lambda x: x[2] == True, process_results))
    batch_outliers = (list(map(lambda x: x[1], row_outliers)))

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

    return batch_outliers


def create_interval_index(batch_method: str, data: pd.DataFrame, clusters: list[models.Cluster], x_label: str,
                          error_tolerance: float, density_split=1000) -> tuple[np.array, list]:
    """
    Creates the interval-based index (based on the x-axis and y-axis) depending on the provided batch_method.

    Returns:
        A tuple where the first element is the indexes along the x-axis as a Numpy Array and the second element is
        the indexes along the y-axis as a list. If the indexes are not applicable to the batch_method, the resulting
        tuple element is set to None.
    """
    x_index = None
    y_index = None
    if batch_method == "tree_index":
        x_index = create_interval_tree(data, clusters, x_label, error_tolerance)
    elif batch_method == "array_index":
        x_index, y_index = create_range_array_index(data, clusters, x_label, error_tolerance)
    elif batch_method == "binary_search":
        pass

    return x_index, y_index


"""
================================
Batch Method: Array Index
================================
"""


def create_range_array_index(data: pd.DataFrame, clusters: np.array, x_label: str, error_tolerance: float,
                             density_split=1000) -> tuple[np.array, list]:
    """
    Creates an indexed range array based on the provided x-values and clusters.

    Args:
        data (DataFrame): The dataset containing x and y values.
        clusters (list): A list of cluster objects with a model that predicts y-values for given x-values.
        x_label (str): The label for the x-values in the dataset.
        error_tolerance (float): The allowable error range for y-values.
        density_split (int, optional): The number of intervals to divide the data into. Default is 1000.

    Returns:
        tuple:
            np.array: The array of x_min values, used for x-axis partitioning.
            list: A list of tuples, where each tuple contains:
                - np.array: The array of y_min values for each cluster.
                - np.array: The array of y_max values for each cluster.
                - list: A list of clusters corresponding to the y-values.
    """
    # Sort the data based on x_label
    sorted_data = data.sort_values(by=x_label)

    # Calculate the number of points per interval
    total_points = len(sorted_data)
    points_per_interval = total_points // density_split  # Divide data points evenly into intervals

    x_min_values = []
    y_max_min_values = []
    prev_x_max = None

    # Loop through the sorted_data in chunks based on the calculated number of points per interval
    i = 0
    while i < total_points:
        chunk = sorted_data.iloc[i:i + points_per_interval]
        x_min, x_max = chunk[x_label].min(), chunk[x_label].max()

        # If the x_min and x_max are the same, adjust the range
        while x_min == x_max and i + points_per_interval < total_points:
            i += points_per_interval
            next_chunk = sorted_data.iloc[i:i + points_per_interval]
            x_max = next_chunk[x_label].max()

        # Edge case where the last interval is still zero-width
        if x_min == x_max:
            x_max += 0.1

        # Ensure continuity by connecting the current x_min with the previous x_max
        if prev_x_max is not None and x_min < prev_x_max:
            x_min = prev_x_max

        x_min_values.append(x_min)
        prev_x_max = x_max

        y_min_list = []
        y_max_list = []
        cluster_list = []

        # Calculate y intervals for each cluster
        for cluster in clusters:
            # Predict y-values for x_min and x_max using the cluster's model
            y_value_at_x_min = cluster.model.predict([[x_min]])[0]
            y_value_at_x_max = cluster.model.predict([[x_max]])[0]

            # Determine slope and adjust y-values accordingly
            if y_value_at_x_max > y_value_at_x_min:  # Positive slope
                y_min = y_value_at_x_max * (1 - error_tolerance)
                y_max = y_value_at_x_min * (1 + error_tolerance)
            else:  # Negative slope
                y_min = y_value_at_x_min * (1 - error_tolerance)
                y_max = y_value_at_x_max * (1 + error_tolerance)

            # Add the y-interval for this cluster to the interval tree
            if y_min < y_max:
                y_min_list.append(y_min)
                y_max_list.append(y_max)
                cluster_list.append(cluster)
            else:
                pass

        # Move to the next chunk
        i += points_per_interval
        y_min_array = np.array(y_min_list)
        y_max_array = np.array(y_max_list)
        y_max_min_values.append((y_min_array, y_max_array, cluster_list))

        x_min_values.append(x_max)

    # Return x_min_values as a numpy array if needed
    return np.array(x_min_values), y_max_min_values


def interval_index_check_array(x_value: float, y_value: float, value_index: int, x_min_array: np.array,
                               y_max_min_values: list, verbose=False) -> bool:
    """
    Checks if the provided x and y values fall within the specified range intervals.

    Args:
        x_value (float): The x-value to check.
        y_value (float): The y-value to check.
        value_index (int): The index of the value to track.
        x_min_array (np.array): Array of x_min values defining x-axis intervals.
        y_max_min_values (list): List of tuples containing y_min, y_max arrays, and corresponding clusters.
        verbose (bool, optional): Whether to print additional details. Default is False.

    Returns:
        bool: True if the x and y values are found within the defined intervals, False otherwise.
    """
    index = find_x_index(x_min_array, x_value)

    # Edge Case: If the value is not found or there are no existing y-clusters at the specified x-range
    if index == -1 or index >= len(y_max_min_values):
        return False

    y_info = y_max_min_values[index]
    y_min_array = y_info[0]
    y_max_array = y_info[1]
    cluster_list = y_info[2]

    cluster_index = find_y_index(y_min_array, y_max_array, y_value)

    if cluster_index == -1:
        if verbose:
            print("not found in y arrays")
        return False

    cluster = cluster_list[cluster_index]

    clusters_lock.acquire()
    try:
        cluster.size += 1
    finally:
        clusters_lock.release()


    return True


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
                            index: int) -> bool:
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

    cluster, predicted_y_val = find_exact_cluster(sorted_clusters, x_value, y_value, error_tolerance)

    if cluster is not None:
        cluster.add_new_value(index, y_value, predicted_y_val)
        return True

    return False


def find_exact_cluster(sorted_clusters: list, x_value: float, y_value: float, error_tolerance: float) -> tuple[
                                                                                                             models.Cluster, float] | \
                                                                                                         tuple[
                                                                                                             None, float]:
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
        return None, 0  # Return None instead of False for consistency

    # Calculate tolerance range for the y_value
    lower_bound = y_value * (1 - error_tolerance)
    upper_bound = y_value * (1 + error_tolerance)

    # Step 2: Start checking from the found index and around it
    n = len(sorted_clusters)

    # Check the cluster at the found index
    is_within_threshold, predicted_y = check_model_within_threshold(sorted_clusters[index], x_value, y_value,
                                                                    error_tolerance)
    if is_within_threshold:
        return sorted_clusters[index], predicted_y

    # Step 3: If not within bounds, explore surrounding clusters
    # Check previous clusters (leftwards in the sorted list)
    for i in range(index - 1, -1, -1):
        is_within_threshold, predicted_y = check_model_within_threshold(sorted_clusters[i], x_value, y_value,
                                                                        error_tolerance)
        if predicted_y > upper_bound:
            # We've gone too far left, no need to check further
            break
        if is_within_threshold:
            return sorted_clusters[i], predicted_y

    # Check next clusters (rightwards in the sorted list)
    for i in range(index + 1, n):
        is_within_threshold, predicted_y = check_model_within_threshold(sorted_clusters[i], x_value, y_value,
                                                                        error_tolerance)
        if predicted_y < lower_bound:
            # We've gone too far right, no need to check further
            break
        if is_within_threshold:
            return sorted_clusters[i], predicted_y

    # Step 4: If no suitable cluster is found, return None and a default predicted value
    return None, 0


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


def check_model_within_threshold(cluster: models.Cluster, x_value: float, y_value: float, error_tolerance: float) -> \
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
    predicted_y = cluster.model.predict([[x_value]])[0]  # Predict y_value using the cluster's model
    lower_bound = y_value * (1 - error_tolerance)
    upper_bound = y_value * (1 + error_tolerance)

    # Return both the check result and the predicted_y
    return lower_bound <= predicted_y <= upper_bound, predicted_y


"""
================================
Batch Method: Tree Index
================================
"""


def interval_index_check(x_value: float, y_value: float, interval_tree: IntervalTree, verbose=False) -> bool:
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
        return False

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
            return True

    # Step 5: If no matching y_interval was found, return False
    total_end = time.perf_counter()
    if verbose:
        print(f"Total time in interval_index_check: {(total_end - total_start) * 1e6:.2f} µs")
        print(f"Time spent in x-interval search: {(x_search_end - x_search_start) * 1e6:.2f} µs")

    return False


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
