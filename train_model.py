import math
import pandas as pd
import sys

from config import config_get
from clustering import k_means, distribution, db_scan, minhal_split, birch, bisecting_kmeans

from outlier_detection import outlier_detection
from models import train_and_evaluate_model, Cluster
from helper import get_recovered_accuracy, percent_difference, size, mse_metrics
from preprocess import create_clusters_for_frequent_items

results_location = config_get('result_location') + config_get('machine_learning_model') + "/"


def train_no_cluster_outliers(output: pd.DataFrame, data: pd.DataFrame, x: str, y: str) -> str:
    """
    Train Machine Learning model using no clustering our outlier detection
    """
    model_results = train_and_evaluate_model(data, x, y, 100)
    only_cluster = model_results[0]

    # Decayed table creating
    decayed_table = create_decayed_table(data, [only_cluster], [], y)
    mse_results = mse_metrics(decayed_table['Original_Y_Value'].tolist(), decayed_table['Predicted_Y_Value'].tolist())

    datatype_of_predicted_attribute = data.dtypes[y]
    size_stats = size(1, 0, datatype_of_predicted_attribute, len(data))

    new_row = {'clustering_method': 'No Clustering/Outliers', 'num_clusters': 1, 'num_outliers': 0,
               'size (bytes)': str(size_stats[0]), 'original_size (bytes)': str(size_stats[1]),
               'percentage_of_original_size': str(size_stats[2]) + '%',
               'average_percent_difference': str(percent_difference(decayed_table['Original_Y_Value'].tolist(),
                                                                    decayed_table['Predicted_Y_Value'].tolist())) + '%',
               'mse': mse_results[1],
               'recovered_accuracy': str(get_recovered_accuracy(decayed_table['Original_Y_Value'].tolist(),
                                                                decayed_table['Predicted_Y_Value'].tolist(), 5)) + '%'}
    output.loc[len(output)] = new_row

    # return percentage of original size
    return str(size_stats[2])


def file_name_info(outlier_before: bool, outlier_after: bool, accuracy_threshold: float, clustering_method: str) -> str:
    """
    Return string associated with outlier detection, accuracy threshold and clustering method options
    """
    file_info = ""
    if outlier_before:
        if outlier_after:
            file_info = "both_" + clustering_method
        else:
            file_info = "before_" + clustering_method
    else:
        if outlier_after:
            file_info = "after_" + clustering_method
        else:
            file_info = "" + clustering_method

    if accuracy_threshold:
        file_info += '_threshold'

    return file_info


def train_model(output: pd.DataFrame, data: pd.DataFrame, x: str, y: str, clustering_method: str, outlier_before: bool, outlier_after: bool, accuracy_threshold: float,
                acceptable_threshold: float, planned_clusters: int) -> tuple[list, list]:
    """
    Takes data and runs the model on that data. There will be clustering done on the data using the `clustering_method` passed to the function.
    If `outlier_before` is true the function will do outlier detection before clustering. If `outlier_after` is set to true, the function will
    do outlier detection after the clustering. If `accuracy_threshold` is true the model will train with an acceptable threshold of 5%
    (hardcoded). `planned_clusters` is maximum number of clusters that will be created by the clustering methods.
    """
    file_info = file_name_info(outlier_before, outlier_after, accuracy_threshold, clustering_method)
    original_data = data.copy()
    original_data_length = len(data)
    outliers_all = []
    clusters = []

    # cluster before if specified
    if outlier_before:
        new_data = outlier_detection(data, 3)
        outliers_all.extend(new_data[0]["Index"].tolist())  # add outliers to outlier list
        data = new_data[1]

    match clustering_method:
        case 'KM':
            clustered_data = k_means(data, planned_clusters, x, y)
        case 'DB':
            clustered_data = db_scan(data, 15, x, y)
        case 'Dist':
            clustered_data = distribution(data, planned_clusters, x, y)
        case 'Birch':
            clustered_data = birch(data, planned_clusters, x, y)
        case 'Bisect':
            clustered_data = bisecting_kmeans(data, planned_clusters, x, y)

    # tracks accuracy and length of clusters for statistics

    for i in range(0, len(clustered_data)):
        # Outlier detection after clustering
        if (outlier_after):
            new_data = outlier_detection(clustered_data[i], 3)
            outliers = new_data[0]["Index"].tolist()  # add outliers to outlier list
            current_data = new_data[1]  # All non outlier values
            outliers_all.extend(outliers)

        else:
            current_data = clustered_data[i]

        # If there is one value or less in the data don't train a model on it
        if (len(current_data) <= 1):
            # If there is a single value in this cluster add it to outliers
            if (len(current_data) == 1):
                outliers_all.extend(current_data[0]["Index"].tolist())
        else:
            # model Running
            print(len(clustered_data))
            model_results = train_and_evaluate_model(current_data, x, y, acceptable_threshold)

            current_cluster = model_results[0]
            clusters.append(current_cluster)
            outliers = model_results[1]

            if (accuracy_threshold):
                outliers_all.extend(outliers)

    # Create outlier file
    decayed_table = create_decayed_table(original_data, clusters, outliers_all, y)
    mse_results = mse_metrics(decayed_table['Original_Y_Value'].tolist(), decayed_table['Predicted_Y_Value'].tolist())

    datatype_of_predicted_attribute = data.dtypes[y]
    size_stats = size(len(clusters), len(outliers_all), datatype_of_predicted_attribute, original_data_length)

    # Row to be added to output
    new_row = {'clustering_method': file_info, 'num_clusters': len(clustered_data), 'num_outliers': len(outliers_all),
               'size (bytes)': str(size_stats[0]), 'original_size (bytes)': str(size_stats[1]),
               'percentage_of_original_size': str(size_stats[2]) + '%',
               'average_percent_difference': str(percent_difference(decayed_table['Original_Y_Value'].tolist(),
                                                                    decayed_table['Predicted_Y_Value'].tolist())) + '%',
               'mse': mse_results[1],
               'recovered_accuracy': str(get_recovered_accuracy(decayed_table['Original_Y_Value'].tolist(),
                                                                decayed_table['Predicted_Y_Value'].tolist(),
                                                                acceptable_threshold)) + '%'}
    output.loc[len(output)] = new_row

    # return percentage of original size
    return clusters, outliers_all


def train_model_unsupervised(output: pd.DataFrame, data: pd.DataFrame, x: str, y: str, clustering_method: str, acceptable_threshold: float, min_split_size: int,
                             preprocess_data: bool, split_cluster_size=2) -> tuple[list, list]:
    """
    This function trains a model using unsupervised learning. It creates a ML model and removes values that satisfy the `acceptable_threshold`
    variable (i.e. 5%, 10%, error etc.). If the length of the resulting outliers is greater than `min_split_size`, the data will be 
    clustered into `split_cluster_size` clusters and fed back into a ML model. This is done recursively.
    """
    file_info = "unsupervised_" + clustering_method

    # Used to track outliers and functions
    outliers_all = []
    current_cluster_number = 0
    clusters = []

    # remove all common values and cluster them
    if preprocess_data:
        processed_clusters, data_to_train_on = create_clusters_for_frequent_items(data, y)
        clusters.extend(processed_clusters)
    else:
        data_to_train_on = data.copy()

    def train_and_split(data: pd.DataFrame, depth: int):
        """
        Recursive function that trains a model on `data`. If the outliers are greater than `min_split_size`, than cluster
        and repeat on the clusters. `depth` is used to track number of function calls for graphing and data display purposes.
        """
        nonlocal outliers_all
        nonlocal current_cluster_number
        nonlocal clusters

        if len(data) < 2:
            outliers_all.extend(data["Index"].tolist())
            return

        # Train model. Add resulting cluster to list of clusters
        model_results = train_and_evaluate_model(data, x, y, acceptable_threshold)
        current_cluster = model_results[0]
        clusters.append(current_cluster)

        outliers = model_results[1]

        if len(outliers) + current_cluster.length() != len(data):
            raise ValueError(
                "The sum of outliers length and current_cluster length does not match the length of the data.")

        # Get indices of outlier table and reconstruct table with only those rows present
        data_to_split = data[data['Index'].isin(outliers)]

        try:
            if len(outliers) > min_split_size:
                match clustering_method:
                    case 'KM':
                        cluster = k_means(data_to_split, split_cluster_size, x, y)
                    case 'Dist':
                        cluster = distribution(data_to_split, split_cluster_size, x, y)
                    case 'Minhal':
                        cluster = minhal_split(data_to_split, current_cluster.model, x, y)
                    case 'Birch':
                        cluster = birch(data_to_split, split_cluster_size, x, y)
                    case 'Bisect':
                        cluster = bisecting_kmeans(data_to_split, split_cluster_size, x, y)

                for i in range(0, len(cluster)):
                    current_cluster_number += 1
                    depth = depth + 1
                    train_and_split(cluster[i], depth)
                    depth = depth - 1
            else:
                outliers_all.extend(outliers)
        except RecursionError:
            print(f"[Warning] Exceeded Python's Recursion Limit of {sys.getrecursionlimit()} at depth {depth}. Adding {len(outliers)} outliers and exiting recursive function. Be aware that this may limit results.")
            outliers_all.extend(outliers)

        return

    train_and_split(data_to_train_on, 0)

    datatype_of_predicted_attribute = data.dtypes[y]

    decayed_table = create_decayed_table(data, clusters, outliers_all, y)
    mse_results = mse_metrics(decayed_table['Original_Y_Value'].tolist(), decayed_table['Predicted_Y_Value'].tolist())

    size_stats = size(len(clusters), len(outliers_all), datatype_of_predicted_attribute, len(data))

    # Row to be added to output
    new_row = {'clustering_method': file_info, 'num_clusters': len(clusters), 'num_outliers': len(outliers_all),
               'size (bytes)': str(size_stats[0]), 'original_size (bytes)': str(size_stats[1]),
               'percentage_of_original_size': str(size_stats[2]) + '%',
               'average_percent_difference': str(percent_difference(decayed_table['Original_Y_Value'].tolist(),
                                                                    decayed_table['Predicted_Y_Value'].tolist())) + '%',
               'mse': mse_results[1],
               'recovered_accuracy': str(get_recovered_accuracy(decayed_table['Original_Y_Value'].tolist(),
                                                                decayed_table['Predicted_Y_Value'].tolist(), 5)) + '%'}
    output.loc[len(output)] = new_row

    # return percentage of original size
    return clusters, outliers_all


def create_decayed_table(data: pd.DataFrame, clusters: list[Cluster], outliers: list[int], y_label: str) -> pd.DataFrame:
    """
    Creates a pandas DataFrame containing information about inliers and outliers.

    Args:
        - data (DataFrame): The original dataset containing strs and target variable.
        - clusters (list of Cluster): A list of Cluster objects representing inlier data points.
        - outliers (list of int): A list of indices corresponding to outlier data points.
        - y_label (str): The label of the target variable.

    Returns:
        - df (DataFrame): A pandas DataFrame containing columns for index, original y values, and predicted y values.
                          The DataFrame is sorted by index in ascending order.
                          
    """
    index_list = []
    original_y_list = []
    predicted_y_list = []

    # For clusters
    for cluster in clusters:
        index_list.extend(cluster.inliers)
        original_y_list.extend(cluster.original_y_values)
        predicted_y_list.extend(cluster.predicted_y_values)

    # For outliers
    for index in outliers:
        original_y_list.append(data.loc[data['Index'] == index, y_label].iloc[0])
        predicted_y_list.append(data.loc[data['Index'] == index, y_label].iloc[0])
        index_list.append(index)

    # Create DataFrame
    data = {'Index': index_list, 'Original_Y_Value': original_y_list, 'Predicted_Y_Value': predicted_y_list}
    df = pd.DataFrame(data)
    df.sort_values(by='Index', inplace=True)

    return df
