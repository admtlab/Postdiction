import numpy as np
import os
import pandas as pd
import random
import sys
import tensorflow as tf
import dask.dataframe as dd
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Lock
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from config import config_get, get_column_info_table
from big_data import data_big_with_noise
from feature_selection import select_best_features, create_dictionary_with_predictor
from helper import function_execution_in_milliseconds
from train_model import train_no_cluster_outliers, train_model, train_model_unsupervised
from batching import fit_new_batch, create_interval_index
from helper import size
from preprocess import free_clusters_to_best_compression

# check for filepath argument before config import
if len(sys.argv) != 2:
    print("Please include a path to desired YAML file as follows: python3 pipeline.py <path_to_yaml>")
    sys.exit(1)

column_info_table = get_column_info_table(config_get('database'))
text_limit_training = config_get('text_limit_training')

def main():
    # load random seeds
    random.seed(100)
    np.random.seed(100)
    tf.random.set_seed(100)

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)

    dask_batch_size = None
    runtime_parameters = config_get('runtime_parameters')
    for _, parameters in runtime_parameters.items():
        dask_batch_size = parameters.get('batch_size')

    if dask_batch_size is not None:
        database_file_type = Path(config_get('database')).suffix
        if database_file_type == '.csv':
            data = dd.read_csv(config_get('database'), blocksize=dask_batch_size, assume_missing=True)
        else:
            data = dd.read_parquet(config_get('database'), blocksize=dask_batch_size, index=False)
    else:
        # load data
        data = pd.read_csv(config_get('database'), sep=',')

    if 'Index' not in data:
        data = data.assign(Index=pd.Series(range(0, len(data))))

    # Expand data if necessary
    if config_get('expand_data_multiplier') > 1:
        data = data_big_with_noise(data, config_get('expand_data_multiplier'), config_get('expanded_file_name'))

    planned_clusters = config_get('planned_clusters')
    inlier_cluster_location = Path('.') / config_get('result_location') / 'clustered_values' / 'inliers.csv'

    data['all_zeroes'] = 0

    overall_columns = ['clustering_method', 'error_threshold', 'min_split_size', 'columns_decayed',
                       'num_outliers', 'size (bytes)', 'original_size (bytes)', 'percentage_of_original_size',
                       'time_elapsed (ms)', 'batch_method', 'batch_size']

    individual_run_columns = ['clustering_method', 'error_threshold', 'min_split_size', 'num_clusters', "num_outliers",
                              'size (bytes)',
                              'original_size (bytes)', 'percentage_of_original_size', 'average_percent_difference',
                              'average_cosine_similarity',
                              'median_cosine_similarity', 'minimum_cosine_similarity', 'mse', 'time_elapsed (ms)',
                              'recovered_accuracy']

    overall_results_df = pd.DataFrame(columns=overall_columns)

    # Extract runtime plans from the YAML file and execute them
    # runtime_parameters = config_get('runtime_parameters')
    for choice, parameters in runtime_parameters.items():
        clustering = parameters.get('clustering')
        cluster_alg = parameters.get('cluster_alg')
        planned_clusters = parameters.get('planned_clusters') if 'planned_clusters' in parameters else planned_clusters
        accuracy = parameters.get('accuracy')
        split_size = parameters.get('split_size')
        outlier_before = parameters.get('outlier_before')
        outlier_after = parameters.get('outlier_after')
        accuracy_tuning = parameters.get('accuracy_tuning')
        binary = parameters.get('binary')
        vector_size = parameters.get('vector_size')
        vector_size = vector_size or 1
        preprocess_data = parameters.get('preprocess_data')
        postprocess_data = parameters.get('postprocess_data')
        predictor = parameters.get('predictor')
        batch_size = parameters.get('batch_size')
        batch_method = parameters.get('batch_method')
        sample_max_attempts = config_get('sample_max_attempts')

        # Used for final time stat, updated if needed
        fitting_time = 0
        postprocess_time = 0

        # Initialize the output DataFrame with the specified columns (will be wiped every iteration)
        individual_run_df = pd.DataFrame(columns=individual_run_columns)

        # Run automate feature selection or use set predictor
        if config_get('automate_feature_selection'):
            print("\n\nSelecting best Features:", )
            sampling_proportion = 1 / (len(data) / 5000)  # Automatically using 5000 input values
            xy_pairs = select_best_features(data, clustering, cluster_alg, accuracy, split_size, outlier_before,
                                            outlier_after, accuracy_tuning, planned_clusters, preprocess_data,
                                            postprocess_data, sampling_proportion)
        elif config_get('use_one_predictor'):
            xy_pairs = create_dictionary_with_predictor(predictor)
        else:
            # In the case where all the predictor AFDs are provided
            xy_pairs = config_get('predicted_by')



        # Train models and get clusters    
        for y, x_list in xy_pairs.items():
            x = x_list[0]

            # Sample from the whole dataset or use whole dataset. Moved to each combination as sampling may change
            # per mapping if Nulls or NANs are present in the data
            if batch_size is not None:
                desired_num_rows = config_get('approximate_sampled_rows')
                sampling_seed = config_get('sampling_seed')
                total_rows_in_data = len(data)
                approx_sample_percentage_per_partition = desired_num_rows / total_rows_in_data
                if approx_sample_percentage_per_partition > 1.0:
                    approx_sample_percentage_per_partition = 1.0
                print('Sampling the partitioned dataset for determining clusters')
                valid_sample = False
                cur_sample_attempt = 0
                cur_data: pd.DataFrame

                # Iteration allows retrials of sampling in the event that no rows are sampled or all have Nulls
                while not valid_sample and cur_sample_attempt < sample_max_attempts:
                    sampled_data = data.sample(frac=approx_sample_percentage_per_partition, random_state=(sampling_seed + cur_sample_attempt))
                    cur_data = sampled_data.compute()
                    print(f'Dropping nulls found in: [{x}, {y}]')
                    cur_data = cur_data.dropna(axis=0, subset=[x, y])

                    if not cur_data.empty:
                        valid_sample = True

                        cur_data = cur_data.reset_index(drop=True)
                        print('Finished sampling, proceeding to calculate clusters')
                    else:
                        print('Empty sample due to Null Values, attempting to resample.')

                    cur_sample_attempt += 1
                if cur_data.empty:
                    print(f'Unable to sample a non-empty dataset after {sample_max_attempts} attempts. Exiting program.')
                    print('Consider trying a different seed, increasing the number of attempts, and analyzing the dataset.')
            else:
                cur_data = data

            if x_list[0] != 'all_zeroes' and column_info_table[x][0] == 'string' and column_info_table[y][
                0] == 'string':
                raise Exception("String data not supported in this version of the pipeline.")
            elif clustering == "supervised":
                results, train_time = function_execution_in_milliseconds(train_model, individual_run_df, cur_data, x, y,
                                                                         cluster_alg, outlier_before, outlier_after,
                                                                         accuracy_tuning, accuracy, planned_clusters)
            elif clustering == "unsupervised":
                results, train_time = function_execution_in_milliseconds(train_model_unsupervised, individual_run_df,
                                                                         cur_data, x, y, cluster_alg, accuracy,
                                                                         split_size, preprocess_data)
            else:
                results, train_time = function_execution_in_milliseconds(train_no_cluster_outliers, individual_run_df,
                                                                         cur_data, x, y)

            clusters = results[0]

            # Since the clusters are sampled from the entire dataset, outliers should be ignored and will be computed
            # as each partition is parsed
            num_outliers = 0

            individual_run_df.at[individual_run_df.index[-1], 'time_elapsed (ms)'] = train_time
            individual_run_df.at[individual_run_df.index[-1], 'min_split_size'] = split_size
            individual_run_df.at[individual_run_df.index[-1], 'error_threshold'] = accuracy

            if batch_size is not None:
                x_index, y_index = create_interval_index(batch_method, cur_data, clusters, x, accuracy)

                with open(Path('.') / config_get('result_location') / 'outliers.txt', 'w') as outliers_file:
                    outliers_file.write('')

                for cur_partition_num in tqdm(range(0, data.npartitions), desc="Processing partitions", ascii=' =',
                                              leave=False):
                    # Note that compute() forces Dask to evaluate the partition and converts to Pandas DataFrame
                    cur_batch = data.get_partition(cur_partition_num).compute()
                    holder, fitting_time = function_execution_in_milliseconds(fit_new_batch, cur_batch, batch_method,
                                                                              clusters, x, y, x_index,
                                                                              y_index, accuracy,
                                                                              clustering_destination=inlier_cluster_location,
                                                                              cur_partition_num=cur_partition_num)
                    outlier_output = list(map(lambda cur_outlier: f'{cur_outlier}\n', holder))
                    with open(Path('.') / config_get('result_location') / 'outliers.txt', 'a') as outliers_file:
                        outliers_file.writelines(outlier_output)

                    num_outliers += len(holder)

            if postprocess_data:
                [clusters, new_num_outliers], postprocess_time = function_execution_in_milliseconds(
                    free_clusters_to_best_compression, clusters, num_outliers, data.dtypes[y], len(data), Path('.') / config_get('result_location') / 'outliers.txt')
                num_outliers = new_num_outliers

            datatype_of_predicted = data.dtypes[y]
            size_stats = size(len(clusters), num_outliers, datatype_of_predicted, len(data))
            individual_run_df.at[
                individual_run_df.index[-1], 'time_elapsed (ms)'] = train_time + fitting_time + postprocess_time
            individual_run_df.at[individual_run_df.index[-1], 'size (bytes)'] = size_stats[0]
            individual_run_df.at[individual_run_df.index[-1], 'original_size (bytes)'] = size_stats[1]
            individual_run_df.at[individual_run_df.index[-1], 'percentage_of_original_size'] = size_stats[2]
            individual_run_df.at[individual_run_df.index[-1], 'num_outliers'] = num_outliers

            compressed_size = individual_run_df['size (bytes)'].astype(int).sum()
            original_size = individual_run_df['original_size (bytes)'].astype(int).sum()
            percentage = (compressed_size / original_size) * 100

            new_row = {'predicting_feature': x_list[0],
                       'clustering_method': individual_run_df.loc[0, 'clustering_method'],
                       'error_threshold': individual_run_df.loc[0, 'error_threshold'],
                       'min_split_size': individual_run_df.loc[0, 'min_split_size'],
                       'columns_decayed': len(individual_run_df),
                       'num_outliers': individual_run_df['num_outliers'].astype(int).sum(),
                       'size (bytes)': compressed_size,
                       'original_size (bytes)': original_size,
                       'percentage_of_original_size': f"{percentage:.4f}%",
                       'time_elapsed (ms)': individual_run_df['time_elapsed (ms)'].astype(float).sum(),
                       'batch_method': batch_method,
                       'batch_size': batch_size}

            overall_results_df.loc[len(overall_results_df)] = new_row

            print("\n\nResults of most recent run:")
            print(individual_run_df)

        print(overall_results_df)
        write_out_results(overall_results_df, individual_run_df)


def write_out_results(overall_results_df: pd.DataFrame, individual_run_df: pd.DataFrame) -> None:
    """
    Write out results of most recent individual run and overall results
    """
    results_location = config_get('result_location') + config_get('machine_learning_model') + "/"
    create_file_if_not_exists(results_location)
    if os.path.exists(results_location + 'output_summary.csv'):
        overall_results_df.to_csv(results_location + '/output_summary.csv', mode='a', header=True, index=False)
    else:
        overall_results_df.to_csv(results_location + '/output_summary.csv', mode='w', header=True, index=False)
    # Output detailed data
    if os.path.exists(results_location + '/output_detailed.csv'):
        individual_run_df.to_csv(results_location + '/output_detailed.csv', mode='a', header=True, index=False)
    else:
        individual_run_df.to_csv(results_location + '/output_detailed.csv', mode='w', header=True, index=False)


def create_file_if_not_exists(file_path: str) -> None:
    """
    Creates a file and its directory if it doesn't exist. Useful for results.
    """
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)


if __name__ == "__main__":
    main()
