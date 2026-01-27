import pandas as pd

from models import Cluster
from helper import size
import math


def values_with_min_freq(data: pd.DataFrame, column_name: str, minimum_frequency: int) -> dict[float, int]:
    """
    Analyzes a specified column of a DataFrame and returns a dictionary
    of values and their frequencies for values that appear minimum_frequency or more times.
    
    :param data: Pandas DataFrame to analyze
    :param column_name: Name of the column to analyze
    :param minimum_frequency: Minimum frequency threshold
    :return: Dictionary of values that appear minimum_frequency or more times and their frequencies
    """
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Calculate value frequencies
    value_counts = data[column_name].value_counts()

    # Filter based on the threshold x
    filtered_counts = value_counts[value_counts >= minimum_frequency]

    # Convert to dictionary and return
    return filtered_counts.to_dict()


def create_clusters_for_frequent_items(data: pd.DataFrame, column_name: str, minimum_frequency=20) -> tuple[
    list[Cluster], pd.DataFrame]:
    """
    Creates clusters for frequently occurring items in a specified column of a DataFrame.
    
    :param data: Pandas DataFrame containing the data
    :param column_name: Name of the column to analyze
    :param minimum_frequency: Minimum frequency threshold for considering an item as frequent (default is 20)
    :return: List of clusters, where each cluster represents a frequently occurring item
    """
    filter_counts = values_with_min_freq(data, column_name, minimum_frequency)
    clusters = []
    for key, value in filter_counts.items():
        inliers = data[data[column_name] == key]['Index'].tolist()
        original_value_list = [key] * value
        current_cluster = Cluster(None, inliers, original_value_list, original_value_list, key)
        clusters.append(current_cluster)

    # Collect rows not put into any cluster
    clustered_indices_set = set(index for cluster in clusters for index in cluster.inliers)
    unclustered_indices = [index for index in data['Index'] if index not in clustered_indices_set]
    unclustered_data = data[data['Index'].isin(unclustered_indices)]

    return clusters, unclustered_data


def free_clusters_to_nearest_power_of_2(clusters: list[Cluster]) -> tuple[list[Cluster], list[float]]:
    # Edge Case: If no clusters were detected, this should return the empty list and avoid rounding
    if len(clusters) <= 1:
        return clusters, []

    sorted_clusters = sorted(clusters.copy())
    nearest_power_of_2 = 2 ** (math.floor(math.log2(len(clusters))))
    new_outliers = []

    for cluster in sorted_clusters:
        # Check to see if we are at the nearest power of 2
        if len(sorted_clusters) == nearest_power_of_2 - 1:
            break

        new_outliers.extend(cluster.inliers)
        sorted_clusters.remove(cluster)

    return sorted_clusters, new_outliers


def free_clusters_to_best_compression(clusters: list[Cluster], num_outliers: int,
                                      datatype_of_predicted_attribute: str, data_length: int, result_location) -> tuple[
    list[Cluster], int]:
    prev_clusters = clusters.copy()
    prev_outliers = num_outliers

    # Edge Case: Check that cluster size can be reduced. If fewer than 2 clusters exist, this step should be skipped
    if len(clusters) <= 1:
        return prev_clusters, prev_outliers

    while (True):
        curr_clusters, curr_outliers = free_clusters_to_nearest_power_of_2(prev_clusters)
        curr_num_outliers = len(curr_outliers) + prev_outliers
        outlier_output = list(map(lambda cur_outlier: f'{cur_outlier}\n', curr_outliers))
        with open(result_location, 'a') as outliers_file:
            outliers_file.writelines(outlier_output)

        prev_compressed_size, original_size, ratio = size(len(prev_clusters), prev_outliers,
                                                          datatype_of_predicted_attribute, data_length)
        curr_compressed_size, original_size, ratio = size(len(curr_clusters), curr_num_outliers,
                                                          datatype_of_predicted_attribute, data_length)

        if curr_compressed_size > prev_compressed_size or len(curr_clusters) <= 1:
            # Reducing to this number of clusters adds too many outliers to be worth it
            break

        prev_clusters = curr_clusters
        prev_outliers = curr_num_outliers

    return prev_clusters, prev_outliers
