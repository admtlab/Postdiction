import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import statistics
import nltk
from multiprocessing import Pool

from config import config_get
from clustering import k_means, distribution, db_scan, minhal_split, birch, bisecting_kmeans, multivariable_k_means, \
    multivariable_minhal_split

from helper import median_cosine_similarity, minimum_cosine_similarity, size_text
from outlier_detection import outlier_detection
from text import vectorize_column_bin, vectorize_qwen3
from models import train_and_evaluate_model, Cluster, multivariable_train_and_evaluate_model
from helper import get_recovered_accuracy, percent_difference, size, mse_metrics, avg_cosine_similarity
from text import get_word2vec_model, vectorize_column
from preprocess import create_clusters_for_frequent_items

results_location = config_get('result_location') + config_get('machine_learning_model') + "/"
column_index_variable = config_get('index_column_name')


def train_no_cluster_outliers(output: pd.DataFrame, data: pd.DataFrame, x: str, y: str,
                              model_type='linear_regression') -> str:
    """
    Train Machine Learning model using no clustering our outlier detection
    """
    model_results = train_and_evaluate_model(data, x, y, 100)
    only_cluster = model_results[0]

    # Decayed table creating
    decayed_table = create_decayed_table(data, [only_cluster], [], y)
    mse_results = mse_metrics(decayed_table['Original_Y_Value'].tolist(), decayed_table['Predicted_Y_Value'].tolist())

    datatype_of_predicted_attribute = data.dtypes[y]
    size_stats = size(1, 0, datatype_of_predicted_attribute, len(data), y)

    new_row = {'predicting_feature': x,
               'predicted_feature': y, 'ml_method': model_type, 'clustering_method': 'No Clustering/Outliers',
               'num_clusters': 1, 'num_outliers': 0,
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


def train_model(output: pd.DataFrame, data: pd.DataFrame, x: str, y: str, clustering_method: str, outlier_before: bool,
                outlier_after: bool, accuracy_threshold: float,
                acceptable_threshold: float, planned_clusters: int, model_type='linear_regression') -> tuple[
    list, list]:
    """
    Takes data and runs the model on that data. There will be clustering done on the data using the `clustering_method` passed to the function.
    If `outlier_before` is true the function will do outlier detection before clustering. If `outlier_after` is set to true, the function will
    do outlier detection after the clustering. If `accuracy_threshold` is true the model will train with an acceptable threshold of 5%
    (hardcoded). `planned_clusters` is maximum number of clusters that will be created by the clustering methods.
    """
    file_info = file_name_info(outlier_before, outlier_after, accuracy_threshold, clustering_method)
    original_data = data.copy()
    original_data_length = len(data)
    # outliers_all = []
    clusters = []

    # cluster before if specified
    if outlier_before:
        new_data = outlier_detection(data, 3)
        # outliers_all.extend(new_data[0][column_index_variable].tolist())  # add outliers to outlier list
        data = new_data[1]

    cluster_centers = [None for _ in range(0, planned_clusters)]

    match clustering_method:
        case 'KM':
            clustered_data, cluster_centers = k_means(data, planned_clusters, x, y)
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
            outliers = new_data[0][column_index_variable].tolist()  # add outliers to outlier list
            current_data = new_data[1]  # All non outlier values
            # outliers_all.extend(outliers)

        else:
            current_data = clustered_data[i]

        # If there is one value or less in the data don't train a model on it
        # if (len(current_data) <= 1):
        #     # If there is a single value in this cluster add it to outliers
        #     pass
        #     # if (len(current_data) == 1):
        #     #     outliers_all.extend(current_data[0][column_index_variable].tolist())
        if len(current_data) > 1:
            model_results = train_and_evaluate_model(current_data, x, y, acceptable_threshold, cluster_index=i,
                                                     representative_point=cluster_centers[i])

            current_cluster = model_results[0]
            clusters.append(current_cluster)
            # outliers = model_results[1]

            # if (accuracy_threshold):
            #     outliers_all.extend(outliers)

    # # Create outlier file
    # decayed_table = create_decayed_table(original_data, clusters, outliers_all, y)
    # mse_results = mse_metrics(decayed_table['Original_Y_Value'].tolist(), decayed_table['Predicted_Y_Value'].tolist())
    #
    # datatype_of_predicted_attribute = data.dtypes[y]
    # size_stats = size(len(clusters), len(outliers_all), datatype_of_predicted_attribute, original_data_length, y)
    #
    # # Row to be added to output
    # new_row = {'predicting_feature': x,
    #            'predicted_feature': y, 'ml_method': model_type, 'clustering_method': file_info,
    #            'num_clusters': len(clustered_data), 'num_outliers': len(outliers_all),
    #            'size (bytes)': str(size_stats[0]), 'original_size (bytes)': str(size_stats[1]),
    #            'percentage_of_original_size': str(size_stats[2]) + '%',
    #            'average_percent_difference': str(percent_difference(decayed_table['Original_Y_Value'].tolist(),
    #                                                                 decayed_table['Predicted_Y_Value'].tolist())) + '%',
    #            'mse': mse_results[1],
    #            'recovered_accuracy': str(get_recovered_accuracy(decayed_table['Original_Y_Value'].tolist(),
    #                                                             decayed_table['Predicted_Y_Value'].tolist(),
    #                                                             acceptable_threshold)) + '%'}
    # output.loc[len(output)] = new_row

    # return percentage of original size
    return clusters, []


def train_model_unsupervised(output: pd.DataFrame, data: pd.DataFrame, x: str, y: str, clustering_method: str,
                             acceptable_threshold: float, min_split_size: int,
                             preprocess_data: bool, split_cluster_size=2, model_type='linear_regression') -> tuple[
    list, list]:
    """
    This function trains a model using unsupervised learning. It creates a ML model and removes values that satisfy the `acceptable_threshold`
    variable (i.e. 5%, 10%, error etc.). If the length of the resulting outliers is greater than `min_split_size`, the data will be 
    clustered into `split_cluster_size` clusters and fed back into a ML model. This is done recursively.
    """
    file_info = "unsupervised_" + clustering_method

    # Used to track outliers and functions
    # outliers_all = []
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
        Iterative replacement for the recursive train_and_split().
        Uses an explicit stack of (dataframe, depth) to avoid recursion overhead.
        """
        # nonlocal outliers_all
        nonlocal current_cluster_number
        nonlocal clusters
        max_depth = config_get('max_unsupervised_depth')

        stack = [(data, depth)]
        cluster_index = 0

        while stack:
            cur_data, cur_depth = stack.pop()

            # If current cluster has less than 2 elements in it, add it to outliers rather than training a model for it
            if len(cur_data) < 2:
                # outliers_all.extend(cur_data[column_index_variable].tolist())
                continue

            if max_depth is not None and current_cluster_number > max_depth:
                break

            # Train model. Add resulting cluster to list of clusters
            model_results = train_and_evaluate_model(cur_data, x, y, acceptable_threshold, cluster_index=cluster_index)
            current_cluster = model_results[0]
            clusters.append(current_cluster)
            cluster_index += 1

            outliers = model_results[1]

            if len(outliers) + current_cluster.length() != len(cur_data):
                raise ValueError(
                    f"The sum of outliers length and current_cluster length does not match the length of the data."
                    f"Outliers: {len(outliers)}, Clusters Length: {current_cluster.length()}, Data Length: {len(cur_data)}")

            # Reconstruct table with only outlier rows
            data_to_split = cur_data[cur_data[column_index_variable].isin(outliers)]

            if len(outliers) > min_split_size:
                match clustering_method:
                    case 'KM':
                        subclusters, _ = k_means(data_to_split, split_cluster_size, x, y)
                    case 'Dist':
                        subclusters = distribution(data_to_split, split_cluster_size, x, y)
                    case 'Minhal':
                        subclusters = minhal_split(data_to_split, current_cluster.model, x, y)
                    case 'Birch':
                        subclusters = birch(data_to_split, split_cluster_size, x, y)
                    case 'Bisect':
                        subclusters = bisecting_kmeans(data_to_split, split_cluster_size, x, y)

                # push subclusters onto stack for further processing (depth-first, mirroring previous recursion)
                for sub in subclusters:
                    current_cluster_number += 1
                    stack.append((sub, cur_depth + 1))
            # else:
            #     outliers_all.extend(outliers)

        return

    train_and_split(data_to_train_on, 0)

    # datatype_of_predicted_attribute = data.dtypes[y]
    #
    # decayed_table = create_decayed_table(data, clusters, outliers_all, y)
    # mse_results = mse_metrics(decayed_table['Original_Y_Value'].tolist(), decayed_table['Predicted_Y_Value'].tolist())
    #
    # size_stats = size(len(clusters), len(outliers_all), datatype_of_predicted_attribute, len(data), y)
    #
    # # Row to be added to output
    # new_row = {'predicting_feature': x,
    #            'predicted_feature': y, 'ml_method': model_type, 'clustering_method': file_info,
    #            'num_clusters': len(clusters), 'num_outliers': len(outliers_all),
    #            'size (bytes)': str(size_stats[0]), 'original_size (bytes)': str(size_stats[1]),
    #            'percentage_of_original_size': str(size_stats[2]) + '%',
    #            'average_percent_difference': str(percent_difference(decayed_table['Original_Y_Value'].tolist(),
    #                                                                 decayed_table['Predicted_Y_Value'].tolist())) + '%',
    #            'mse': mse_results[1],
    #            'recovered_accuracy': str(get_recovered_accuracy(decayed_table['Original_Y_Value'].tolist(),
    #                                                             decayed_table['Predicted_Y_Value'].tolist(), 5)) + '%'}
    # output.loc[len(output)] = new_row

    # return percentage of original size
    return clusters, []


def _unsupervised_textual_clusters(output: pd.DataFrame, data: pd.DataFrame, x: str, y: str,
                                   acceptable_threshold: float, min_split_size: int, split_cluster_size=2,
                                   model_type='linear_regression', accuracy_metric="cosine"):
    # outliers_all = []
    clusters = []
    current_cluster_number = 0

    max_depth = config_get('max_unsupervised_depth')

    def train_and_split(data):
        nonlocal clusters, current_cluster_number

        stack = [(data, 0)]
        cluster_index = 0

        while stack:
            cur_data, cur_depth = stack.pop()

            if len(cur_data) < 2:
                # outliers_all.extend(cur_data[column_index_variable].tolist())
                continue

            if current_cluster_number > max_depth:
                break

            feature_len = len(cur_data.iloc[0][x])

            X = np.vstack(cur_data[x].values)  # shape (n_samples, n_features)
            representative_point = [statistics.median(X[:, i]) for i in range(X.shape[1])]

            cluster, outliers = train_and_evaluate_model(cur_data, x, y, acceptable_threshold, accuracy_metric,
                                                         predicting_feature_count=feature_len,
                                                         cluster_index=cluster_index,
                                                         representative_point=representative_point)
            clusters.append(cluster)
            cluster_index += 1

            if len(outliers) + cluster.length() != len(cur_data):
                raise ValueError(
                    f"Cluster size mismatch: outliers={len(outliers)}, "
                    f"inliers={cluster.length()}, total={len(cur_data)}"
                )

            data_to_split = cur_data[cur_data[column_index_variable].isin(outliers)]

            if len(outliers) > min_split_size:
                X = np.vstack(data_to_split[x].values)
                median_vec = np.median(X, axis=0)

                dists = np.linalg.norm(X - median_vec, axis=1)
                threshold = np.median(dists)

                sub_A = data_to_split[dists >= threshold]
                sub_B = data_to_split[dists < threshold]

                for sub in [sub_A, sub_B]:
                    current_cluster_number += 1
                    stack.append((sub, cur_depth + 1))
            # else:
            #     outliers_all.extend(outliers)

    train_and_split(data)

    return clusters, []


def _process_single_cluster(args):
    """
    Helper function executed in worker processes for textual training.
    """
    (cluster, x, y, acceptable_threshold, accuracy_metric,
     longest_string_length_x, x_max_length, accuracy_threshold, centroid) = args

    cluster_len = len(cluster)

    # Single-value cluster marked as outlier
    if cluster_len == 1:
        return None, [cluster.iloc[0][column_index_variable]]

    # Empty cluster
    if cluster_len == 0:
        return None, []

    # Train model
    feature_len = longest_string_length_x or x_max_length
    current_cluster, outliers = train_and_evaluate_model(
        cluster, x, y, acceptable_threshold, accuracy_metric,
        predicting_feature_count=feature_len, representative_point=centroid
    )

    if not accuracy_threshold:
        outliers = []

    return current_cluster, outliers


def train_model_text(output: pd.DataFrame, data: pd.DataFrame, x: str, y: str, planned_clusters: int, vector_size=1,
                     accuracy_metric="cosine", clustering_method='KM', outlier_before=False, outlier_after=False,
                     accuracy_threshold=True, acceptable_threshold=0.95, binary=False, model_type='linear_regression',
                     longest_string_length_x=None, longest_string_length_y=1, min_split_size=4, split_cluster_size=2, embed_model='word2vec', embed_dim=32):
    file_info = file_name_info(outlier_before, outlier_after, accuracy_threshold, clustering_method)

    # data[x] = data[x].fillna("").astype(str)
    # data[y] = data[y].fillna("").astype(str)

    clusters = []

    if binary:
        data = vectorize_column_bin(data, x, longest_string_length_x)
        data = vectorize_column_bin(data, y, longest_string_length_y)
        x_word2vec_model = y_word2vec_model = None
        x_word2vec_model_name = y_word2vec_model_name = None
        x_max_length = longest_string_length_x
        y_max_length = longest_string_length_y
    elif embed_model == 'word2vec':
        x_word2vec_model, x_word2vec_model_name = get_word2vec_model(data, x, vector_size)
        y_word2vec_model, y_word2vec_model_name = get_word2vec_model(data, y, vector_size)

        data, x_max_length = vectorize_column(data, x, x_word2vec_model, vector_size, longest_string_length_x)
        data, y_max_length = vectorize_column(data, y, y_word2vec_model, vector_size, longest_string_length_y)
    elif embed_model == 'qwen3':
        x_word2vec_model = y_word2vec_model = None
        x_word2vec_model_name = y_word2vec_model_name = None
        x_max_length = y_max_length = embed_dim
        data = vectorize_qwen3(data, x, embed_dim)
        data = vectorize_qwen3(data, y, embed_dim)

    if outlier_before:
        raise Exception("outlier detection with string data currently not implemented")

    if clustering_method == 'KM':
        clustered_data, cluster_centers = k_means(data, planned_clusters, x, y, binary)

    elif clustering_method == 'Minhal':
        clusters, outliers_all = _unsupervised_textual_clusters(output=output, data=data, x=x, y=y,
                                                                    acceptable_threshold=acceptable_threshold,
                                                                    min_split_size=min_split_size,
                                                                    split_cluster_size=split_cluster_size,
                                                                    model_type=model_type, accuracy_metric=accuracy_metric)
        # Exit here since clusters and outliers are dynamically determined not using supervised clustering
        return clusters, outliers_all, x_word2vec_model, y_word2vec_model
    else:
        raise Exception(f"{clustering_method} clustering not implemented for text")

    worker_args = [
        (cluster, x, y, acceptable_threshold, accuracy_metric, longest_string_length_x, x_max_length,
         accuracy_threshold, centroid[:(-1 * y_max_length)])
        for cluster, centroid in zip(clustered_data, cluster_centers)
    ]

    results = []

    # LSTM models cannot be trained or predicted inside multiprocessing
    if model_type == 'lstm':
        for args in worker_args:
            results.append(_process_single_cluster(args))
    else:
        # Parallelize for linear regression and other CPU-safe models
        with Pool(processes=config_get('num_processes')) as pool:
            results = pool.map(_process_single_cluster, worker_args)

    for current_cluster, outliers in results:
        if current_cluster is not None:
            clusters.append(current_cluster)

    return clusters, [], x_word2vec_model, y_word2vec_model


def multivariable_train_model(output: pd.DataFrame, data: pd.DataFrame, x: list[str], y: str, clustering_method: str,
                              outlier_before: bool, outlier_after: bool, accuracy_threshold: float,
                              acceptable_threshold: float, planned_clusters: int, model_type='multiple LR') -> tuple[
    list, list]:
    """
    Takes data and runs the model on that data. There will be clustering done on the data using the `clustering_method` passed to the function.
    If `outlier_before` is true the function will do outlier detection before clustering. If `outlier_after` is set to true, the function will
    do outlier detection after the clustering. If `accuracy_threshold` is true the model will train with an acceptable threshold of 5%
    (hardcoded). `planned_clusters` is maximum number of clusters that will be created by the clustering methods.
    """
    file_info = file_name_info(outlier_before, outlier_after, accuracy_threshold, clustering_method)
    # original_data = data.copy()
    # original_data_length = len(data)
    # outliers_all = []
    clusters = []

    # cluster before if specified
    if outlier_before:
        new_data = outlier_detection(data, 3)
        # outliers_all.extend(new_data[0][column_index_variable].tolist())  # add outliers to outlier list
        data = new_data[1]

    match clustering_method:
        case 'KM':
            clustered_data = multivariable_k_means(data, planned_clusters, x, y)
        case 'DB':
            # clustered_data = db_scan(data, 15, x, y)
            print('DB currently not implemented as multivariable')
            sys.exit(0)
        case 'Dist':
            # clustered_data = distribution(data, planned_clusters, x, y)
            print('Dist currently not implemented as multivariable')
            sys.exit(0)
        case 'Birch':
            # clustered_data = birch(data, planned_clusters, x, y)
            print('Birch currently not implemented as multivariable')
            sys.exit(0)
        case 'Bisect':
            # clustered_data = bisecting_kmeans(data, planned_clusters, x, y)
            print('Bisect currently not implemented as multivariable')
            sys.exit(0)

    # tracks accuracy and length of clusters for statistics

    for i in range(0, len(clustered_data)):
        # Outlier detection after clustering
        if (outlier_after):
            new_data = outlier_detection(clustered_data[i], 3)
            outliers = new_data[0][column_index_variable].tolist()  # add outliers to outlier list
            current_data = new_data[1]  # All non outlier values
            # outliers_all.extend(outliers)

        else:
            current_data = clustered_data[i]

        # If there is one value or less in the data don't train a model on it
        # if (len(current_data) <= 1):
        #     # If there is a single value in this cluster add it to outliers
        #     if (len(current_data) == 1):
        #         # outliers_all.extend(current_data[0][column_index_variable].tolist())
        if len(current_data) > 1:
            # model Running
            # print(len(clustered_data))
            model_results = multivariable_train_and_evaluate_model(current_data, x, y, acceptable_threshold,
                                                                   cluster_index=i)
            # outliers = model_results[0]

            current_cluster = model_results[0]
            clusters.append(current_cluster)
            outliers = model_results[1]

            # if (accuracy_threshold):
            #     outliers_all.extend(outliers)

    # # Create outlier file
    # decayed_table = create_decayed_table(original_data, clusters, outliers_all, y)
    # mse_results = mse_metrics(decayed_table['Original_Y_Value'].tolist(), decayed_table['Predicted_Y_Value'].tolist())
    #
    # datatype_of_predicted_attribute = data.dtypes[y]
    # size_stats = size(len(clusters), len(outliers_all), datatype_of_predicted_attribute, original_data_length, y)
    #
    # # Row to be added to output
    # new_row = {'predicting_feature': x,
    #            'predicted_feature': y, 'ml_method': model_type, 'clustering_method': file_info,
    #            'num_clusters': len(clustered_data), 'num_outliers': len(outliers_all),
    #            'size (bytes)': str(size_stats[0]), 'original_size (bytes)': str(size_stats[1]),
    #            'percentage_of_original_size': str(size_stats[2]) + '%',
    #            'average_percent_difference': str(percent_difference(decayed_table['Original_Y_Value'].tolist(),
    #                                                                 decayed_table['Predicted_Y_Value'].tolist())) + '%',
    #            'mse': mse_results[1],
    #            'recovered_accuracy': str(get_recovered_accuracy(decayed_table['Original_Y_Value'].tolist(),
    #                                                             decayed_table['Predicted_Y_Value'].tolist(),
    #                                                             acceptable_threshold)) + '%'}
    # output.loc[len(output)] = new_row

    # return percentage of original size
    return clusters, []


def multivariable_train_model_unsupervised(output: pd.DataFrame, data: pd.DataFrame, x: list[str], y: str,
                                           clustering_method: str, acceptable_threshold: float, min_split_size: int,
                                           preprocess_data: bool, split_cluster_size=2, model_type='multiple LR') -> \
        tuple[list, list]:
    """
    This function trains a model using unsupervised learning. It creates a ML model and removes values that satisfy the `acceptable_threshold`
    variable (i.e. 5%, 10%, error etc.). If the length of the resulting outliers is greater than `min_split_size`, the data will be
    clustered into `split_cluster_size` clusters and fed back into a ML model. This is done recursively.
    """
    file_info = "unsupervised_" + clustering_method

    # Used to track outliers and functions
    # outliers_all = []
    current_cluster_number = 0
    clusters = []

    # remove all common values and cluster them
    if preprocess_data:
        processed_clusters, data_to_train_on = create_clusters_for_frequent_items(data, y)
        clusters.extend(processed_clusters)
    else:
        data_to_train_on = data.copy()

    def multivariable_train_and_split(data: pd.DataFrame, depth: int):
        """
        Iterative replacement for recursive multivariable train_and_split.
        Uses an explicit stack of (dataframe, depth) to avoid recursion overhead.
        """
        # nonlocal outliers_all
        nonlocal current_cluster_number
        nonlocal clusters
        max_depth = config_get('max_unsupervised_depth')

        stack = [(data, depth)]
        cluster_index = 0

        while stack:
            cur_data, cur_depth = stack.pop()

            # If current cluster has less than 2 elements in it, add it to outliers rather than training a model for it
            if len(cur_data) < 2:
                # outliers_all.extend(cur_data[column_index_variable].tolist())
                continue

            if current_cluster_number > max_depth:
                break

            # Train model. Add resulting cluster to list of clusters
            model_results = multivariable_train_and_evaluate_model(cur_data, x, y, acceptable_threshold,
                                                                   cluster_index=cluster_index)
            current_cluster = model_results[0]
            clusters.append(current_cluster)
            cluster_index += 1

            outliers = model_results[1]

            if len(outliers) + current_cluster.length() != len(cur_data):
                raise ValueError(
                    f"The sum of outliers length and current_cluster length does not match the length of the data."
                    f"Outliers: {len(outliers)}, Clusters Length: {current_cluster.length()}, Data Length: {len(cur_data)}")

            # Reconstruct table with only outlier rows
            data_to_split = cur_data[cur_data[column_index_variable].isin(outliers)]

            if len(outliers) > min_split_size:
                match clustering_method:
                    case 'KM':
                        subclusters = multivariable_k_means(data_to_split, split_cluster_size, x, y)
                    case 'Dist':
                        # cluster = distribution(data_to_split, split_cluster_size, x, y)
                        print('Dist currently not implemented as multivariable')
                        sys.exit(0)
                    case 'Minhal':
                        subclusters = multivariable_minhal_split(data_to_split, current_cluster.model, x, y)
                    case 'Birch':
                        # cluster = birch(data_to_split, split_cluster_size, x, y)
                        print('Birch currently not implemented as multivariable')
                        sys.exit(0)
                    case 'Bisect':
                        # cluster = bisecting_kmeans(data_to_split, split_cluster_size, x, y)
                        print('Bisecting KMeans currently not implemented as multivariable')
                        sys.exit(0)

                # push subclusters onto stack for further processing (depth-first, mirroring previous recursion)
                for sub in subclusters:
                    current_cluster_number += 1
                    stack.append((sub, cur_depth + 1))
            # else:
            #     outliers_all.extend(outliers)

        return

    multivariable_train_and_split(data_to_train_on, 0)

    # datatype_of_predicted_attribute = data.dtypes[y]
    #
    # decayed_table = create_decayed_table(data, clusters, outliers_all, y)
    # mse_results = mse_metrics(decayed_table['Original_Y_Value'].tolist(), decayed_table['Predicted_Y_Value'].tolist())
    #
    # size_stats = size(len(clusters), len(outliers_all), datatype_of_predicted_attribute, len(data), y,
    #                   predicting_feature_count=len(x))
    #
    # # Row to be added to output
    # new_row = {'predicting_feature': x,
    #            'predicted_feature': y, 'ml_method': model_type, 'clustering_method': file_info,
    #            'num_clusters': len(clusters), 'num_outliers': len(outliers_all),
    #            'size (bytes)': str(size_stats[0]), 'original_size (bytes)': str(size_stats[1]),
    #            'percentage_of_original_size': str(size_stats[2]) + '%',
    #            'average_percent_difference': str(percent_difference(decayed_table['Original_Y_Value'].tolist(),
    #                                                                 decayed_table['Predicted_Y_Value'].tolist())) + '%',
    #            'mse': mse_results[1],
    #            'recovered_accuracy': str(get_recovered_accuracy(decayed_table['Original_Y_Value'].tolist(),
    #                                                             decayed_table['Predicted_Y_Value'].tolist(), 5)) + '%'}
    # output.loc[len(output)] = new_row

    # return percentage of original size
    return clusters, []


def create_decayed_table(data: pd.DataFrame, clusters: list[Cluster], outliers: list[int],
                         y_label: str) -> pd.DataFrame:
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
        # print(data['Index'], index)
        original_y_list.append(data.loc[data[column_index_variable] == index, y_label].iloc[0])
        predicted_y_list.append(data.loc[data[column_index_variable] == index, y_label].iloc[0])
        index_list.append(index)

    # Create DataFrame
    data = {column_index_variable: index_list, 'Original_Y_Value': original_y_list,
            'Predicted_Y_Value': predicted_y_list}
    df = pd.DataFrame(data)
    df.sort_values(by=column_index_variable, inplace=True)

    return df
