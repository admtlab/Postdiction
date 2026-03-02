import numpy as np
import os
import pandas as pd
import random
import sys
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import dask.dataframe as dd
import gc
import psutil
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

from config import config_get, get_column_info_table
from big_data import data_big_with_noise
from feature_selection import select_best_features, create_dictionary_with_predictor
from helper import function_execution_in_milliseconds, create_file_if_not_exists, format_duration
from train_model import train_no_cluster_outliers, train_model, train_model_unsupervised
from batching import fit_new_batch, create_interval_index
from helper import size, append_outlier_records, get_record_size, sort_temp_file_in_chunks, merge_sorted_chunks, \
    parse_byte_size, clear_temp_dir, load_cluster_map_files, \
    persist_cluster
from preprocess import free_clusters_to_best_compression
from analysis import plot_storage_reduction


@dataclass
class RuntimeConfig:
    clustering: str = None
    cluster_alg: str = None
    planned_clusters: Optional[int] = None
    accuracy: float = None
    model_type: str = None
    split_size: int = None
    outlier_before: bool = None
    outlier_after: bool = None
    accuracy_tuning: bool = None
    binary: Any = None
    vector_size: int = 1
    preprocess_data: bool = None
    postprocess_data: bool = None
    predictor: list[str] = None
    batch_size: str = None
    batch_method: str = None
    sample_max_attempts: int = None
    leaf_level: int = 40
    persist_structures: bool = False
    build_analysis: bool = False
    provide_error: bool = False

    @classmethod
    def from_dict(cls, parameters: dict, planned_clusters_default: int, sample_max_attempts: int):
        return cls(
            clustering=parameters.get("clustering"),
            cluster_alg=parameters.get("cluster_alg"),
            planned_clusters=parameters.get("planned_clusters", planned_clusters_default),
            accuracy=parameters.get("accuracy"),
            model_type=parameters.get("machine_learning_model"),
            split_size=parameters.get("split_size"),
            outlier_before=parameters.get("outlier_before"),
            outlier_after=parameters.get("outlier_after"),
            accuracy_tuning=parameters.get("accuracy_tuning"),
            binary=parameters.get("binary"),
            vector_size=parameters.get("vector_size") or 1,
            preprocess_data=parameters.get("preprocess_data"),
            postprocess_data=parameters.get("postprocess_data"),
            predictor=parameters.get("predictor"),
            batch_size=parameters.get("batch_size"),
            batch_method=parameters.get("batch_method"),
            sample_max_attempts=sample_max_attempts,
            leaf_level=parameters.get("leaf_level", 40),
            persist_structures=parameters.get("persist_structures"),
            build_analysis=parameters.get("build_analysis"),
            provide_error=parameters.get("provide_error")
        )


# check for filepath argument before config import
if len(sys.argv) != 2:
    print("Please include a path to desired YAML file as follows: python3 pipeline.py <path_to_yaml>")
    sys.exit(1)


def log_memory_usage(stage: str) -> None:
    """
    Logs the current memory usage at a specified stage for
    debugging purposes.
    Args:
        stage: The current stage when memory is logged.
    """
    if config_get('memory_logging'):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"[{stage}] Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")


def main() -> None:
    # load random seeds
    random.seed(100)
    np.random.seed(100)
    tf.random.set_seed(100)

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)

    column_info_table = get_column_info_table(config_get('database'))

    # check if using batched variant to read dataset (Dask)
    dask_batch_size = None
    runtime_parameters = config_get('runtime_parameters')
    for _, parameters in runtime_parameters.items():
        dask_batch_size = parameters.get('batch_size')

    if dask_batch_size is not None:
        database_file_type = Path(config_get('database')).suffix
        if database_file_type == '.csv':
            data = dd.read_csv(config_get('database'), blocksize=dask_batch_size, assume_missing=True)
        else:
            data = dd.read_parquet(config_get('database'), blocksize=dask_batch_size, split_row_groups='adaptive',
                                   index=False)
    else:
        # load data
        data = pd.read_csv(config_get('database'), sep=',')

    column_index_variable = config_get('index_column_name')
    if column_index_variable not in data.columns:
        data = data.assign(Index=pd.Series(range(0, len(data))))

    # Expand data if necessary
    if config_get('expand_data_multiplier') > 1:
        data = data_big_with_noise(data, config_get('expand_data_multiplier'), config_get('expanded_file_name'))

    planned_clusters = config_get('planned_clusters')
    inlier_cluster_location = None
    data['all_zeroes'] = 0

    overall_columns = ['predicting_feature', 'predicted_feature', 'ml_method', 'clustering_method', 'error_threshold',
                       'min_split_size', 'columns_decayed',
                       'num_outliers', 'size (bytes)', 'original_size (bytes)', 'percentage_of_original_size',
                       'time_elapsed (ms)', 'batch_method', 'batch_size', 'models_size (bytes)', 'outlier_size (bytes)',
                       'original_bits', 'bits', 'mean_outlier_error', 'max_outlier_error', 'min_outlier_error',
                       'median_outlier_error', 'std_outlier_error', 'mean_inlier_error', 'max_inlier_error',
                       'min_inlier_error', 'median_inlier_error', 'std_inlier_error']

    individual_run_columns = ['predicting_feature', 'predicted_feature', 'ml_method', 'clustering_method',
                              'error_threshold', 'min_split_size', 'num_clusters', "num_outliers",
                              'size (bytes)',
                              'original_size (bytes)', 'percentage_of_original_size', 'average_percent_difference',
                              'average_cosine_similarity',
                              'median_cosine_similarity', 'minimum_cosine_similarity', 'mse', 'time_elapsed (ms)',
                              'recovered_accuracy', 'models_size (bytes)', 'outlier_size (bytes)', 'original_bits',
                              'bits']

    overall_results_df = pd.DataFrame(columns=overall_columns)

    use_global_accuracy = config_get('use_global_accuracy')
    use_global_model_type = config_get('use_one_model_type')
    missing_value_methods = config_get('handle_nulls')

    log_memory_usage('After loading the data')

    # Extract runtime plans from the YAML file and execute them
    for choice, parameters in runtime_parameters.items():
        overall_results_df = process_runtime_choice(parameters=parameters, data=data, planned_clusters=planned_clusters,
                                                    individual_run_columns=individual_run_columns,
                                                    overall_results_df=overall_results_df,
                                                    column_index_variable=column_index_variable,
                                                    column_info_table=column_info_table,
                                                    missing_value_methods=missing_value_methods,
                                                    inlier_cluster_location=inlier_cluster_location,
                                                    dask_batch_size=dask_batch_size,
                                                    use_global_model_type=use_global_model_type,
                                                    use_global_accuracy=use_global_accuracy)

    # results_location = Path('.') / config_get('result_location') / 'analysis'
    # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # results_location = results_location / timestamp
    # create_file_if_not_exists(str(results_location))
    # plot_storage_reduction(overall_results_df, results_location)


def process_runtime_choice(parameters, data, planned_clusters: int, individual_run_columns: list[str],
                           overall_results_df: pd.DataFrame, column_index_variable, column_info_table,
                           missing_value_methods, inlier_cluster_location, dask_batch_size: str,
                           use_global_model_type: bool, use_global_accuracy: bool):
    """
    Process a single runtime choice and update overall_results_df.
    Returns updated overall_results_df.
    """

    cfg = RuntimeConfig.from_dict(parameters, planned_clusters_default=planned_clusters,
                                  sample_max_attempts=config_get('sample_max_attempts'))

    individual_run_df = pd.DataFrame(columns=individual_run_columns)

    # -----------------------------
    # Build xy_pairs
    # -----------------------------
    if config_get('automate_feature_selection'):
        print("\n\nSelecting best Features:")
        sampling_proportion = 1 / (len(data) / 5000)

        xy_pairs = select_best_features(data, cfg.clustering, cfg.cluster_alg, cfg.accuracy, cfg.split_size,
                                        cfg.outlier_before, cfg.outlier_after, cfg.accuracy_tuning,
                                        cfg.planned_clusters, cfg.preprocess_data, cfg.postprocess_data,
                                        sampling_proportion, use_global_accuracy, config_get('predicted_by'),
                                        use_single_model_type=use_global_model_type)
    elif config_get('use_one_predictor'):
        xy_pairs = create_dictionary_with_predictor(cfg.predictor)
    else:
        xy_pairs = config_get('predicted_by')

    # -----------------------------
    # Process each target column
    # -----------------------------
    for y in xy_pairs.keys():
        individual_run_df, overall_results_df = process_target_column(y=y, cfg=cfg, xy_pairs=xy_pairs, data=data,
                                                                      individual_run_df=individual_run_df,
                                                                      overall_results_df=overall_results_df,
                                                                      column_index_variable=column_index_variable,
                                                                      column_info_table=column_info_table,
                                                                      missing_value_methods=missing_value_methods,
                                                                      inlier_cluster_location=inlier_cluster_location,
                                                                      dask_batch_size=dask_batch_size,
                                                                      use_global_model_type=use_global_model_type,
                                                                      use_global_accuracy=use_global_accuracy)

        print(overall_results_df)
        write_out_results(overall_results_df, individual_run_df)

    if cfg.persist_structures:
        results_location = Path('.') / config_get('result_location')
        temp_dir = results_location / "outliers_tmp"
        final_outlier_file = results_location / 'outliers' / "all_outliers.bin"
        clusters_temp_dir = results_location / "clusters_tmp"
        final_clusters_dir = results_location / 'final_output/'

        # Make sure the directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)
        clusters_temp_dir.mkdir(parents=True, exist_ok=True)
        final_clusters_dir.mkdir(parents=True, exist_ok=True)

        # Collect all temp files (one per attribute)
        temp_files = sorted(temp_dir.glob("*.tmp"))
        record_size = get_record_size(data)

        # This will hold all chunk file paths
        all_chunks = []

        # Sort each temp file into sorted chunks
        for temp_file in temp_files:
            chunks = sort_temp_file_in_chunks(temp_file, record_size, chunk_size_bytes=parse_byte_size(cfg.batch_size))
            all_chunks.extend(chunks)

        # Merge all sorted chunks into the final global file
        merge_sorted_chunks(all_chunks, final_outlier_file, record_size)
        clear_temp_dir(temp_dir)

        # Merge and combine cluster files
        data = data.reset_index(drop=True)
        data = data.assign(row_index=data.index)
        data = data.set_index('row_index')
        cluster_map = load_cluster_map_files(clusters_temp_dir)
        cluster_map = cluster_map.set_index('row_index')

        for col in cluster_map.columns:
            data[col] = cluster_map[col]
        # final_data = data.merge(cluster_map, on="row_index", how="left")

        data.to_parquet(final_clusters_dir, write_index=False)
        clear_temp_dir(clusters_temp_dir)

    if cfg.build_analysis:
        results_location = Path('.') / config_get('result_location') / 'analysis'
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_location = results_location / timestamp
        create_file_if_not_exists(str(results_location))
        plot_storage_reduction(overall_results_df, individual_run_df, results_location)

    if cfg.provide_error:
        clear_temp_dir(Path('.') / config_get('result_location') / 'errors')

    return overall_results_df


def process_target_column(y: str, cfg: RuntimeConfig, xy_pairs, data, individual_run_df: pd.DataFrame,
                          overall_results_df: pd.DataFrame, column_index_variable: str, column_info_table,
                          missing_value_methods, inlier_cluster_location, dask_batch_size: str,
                          use_global_model_type: bool, use_global_accuracy: bool):
    """
    Process a single target column y and update individual_run_df + overall_results_df.
    Returns updated dataframes.
    """

    # -----------------------------
    # Resolve model/clustering settings
    # -----------------------------
    if not use_global_model_type:
        model_type = xy_pairs[y]['model']
        clustering = xy_pairs[y]['clustering']
        cluster_alg = xy_pairs[y]['cluster_alg']
        batch_method = xy_pairs[y]['batch_method']
        if clustering == 'supervised':
            planned_clusters = xy_pairs[y]['planned_clusters']

    x_list = xy_pairs[y]['predictors'] if not config_get('use_one_predictor') else cfg.predictor
    x = x_list if cfg.model_type == 'multivariable LR' else x_list[0]

    data = data.repartition(partition_size=dask_batch_size)

    if cfg.model_type == 'multivariable LR':
        raise ValueError("Multivariable LR not available in this version.")
    else:
        x_replacement_vals = missing_value_methods[x]['replacement_value']

    if not use_global_accuracy:
        accuracy = xy_pairs[y]['accuracy']

    # -----------------------------
    # Sample or use full dataset
    # -----------------------------
    if cfg.batch_size is not None:
        if cfg.model_type == 'multivariable LR':
            raise ValueError("Multivariable LR not available in this version.")
        else:
            cur_data = single_predictor_batch_sample(data, cfg.sample_max_attempts, column_index_variable, x, y,
                                                     missing_value_methods)
    else:
        if cfg.model_type == 'multivariable LR':
            raise ValueError("Multivariable LR not available in this version.")
        else:
            cur_data = data[[column_index_variable, x, y]]

    # -----------------------------
    # Determine clusters
    # -----------------------------
    results, train_time = determine_clusters(cfg, individual_run_df, cur_data, x, y, x_list, column_info_table)
    clusters = results[0]

    log_memory_usage('After determining clusters')

    num_outliers = 0
    individual_run_df.at[individual_run_df.index[-1], 'time_elapsed (ms)'] = train_time
    individual_run_df.at[individual_run_df.index[-1], 'min_split_size'] = cfg.split_size
    individual_run_df.at[individual_run_df.index[-1], 'error_threshold'] = cfg.accuracy

    # -----------------------------
    # Partition processing
    # -----------------------------
    if cfg.batch_size is not None:
        if cfg.batch_method == 'kd_tree_index':
            assert cfg.model_type == 'multivariable LR'
            raise ValueError("Multivariable LR not available in this version.")

        x_index, y_index, kd_index = create_interval_index(cfg.batch_method, cur_data, clusters, x, cfg.accuracy,
                                                           leaf_level=cfg.leaf_level)

        log_memory_usage('After computing indexes')

        for cur_partition_num in tqdm(range(data.npartitions), desc="Processing partitions", ascii=' =', leave=False):
            outlier_values, holder, fitting_time, outliers_found, cluster_tuple_lst, outlier_errors, inlier_errors = process_partition(
                data=data, cur_partition_num=cur_partition_num,
                cfg=cfg, x=x, y=y,
                column_index_variable=column_index_variable,
                x_index=x_index, y_index=y_index,
                kd_index=kd_index, clusters=clusters,
                inlier_cluster_location=inlier_cluster_location,
                missing_value_methods=missing_value_methods,
                x_replacement_vals=x_replacement_vals)

            num_outliers += outliers_found

            # Write errors to file for later eval
            if cfg.build_analysis:
                outlier_path = Path('.') / config_get('result_location') / 'errors' / f"{y}" / "outliers"
                outlier_path.mkdir(parents=True, exist_ok=True)

                outlier_df = pd.DataFrame({"outlier_errors": outlier_errors})
                outlier_df.to_parquet(
                    outlier_path / f"batch_{cur_partition_num}.parquet",
                    engine="pyarrow",
                    index=False
                )

                inlier_path = Path('.') / config_get('result_location') / 'errors' / f"{y}" / "inliers"
                inlier_path.mkdir(parents=True, exist_ok=True)

                inlier_df = pd.DataFrame({"inlier_errors": inlier_errors})
                inlier_df.to_parquet(
                    inlier_path / f"batch_{cur_partition_num}.parquet",
                    engine="pyarrow",
                    index=False
                )

            if cfg.persist_structures:
                # Write outliers to temporary binary sequence file
                outlier_temp_path = Path('.') / config_get(
                    'result_location') / 'outliers_tmp' / f'temporary_outliers_of_{y}.tmp'
                create_file_if_not_exists(outlier_temp_path)
                append_outlier_records(outlier_values, holder, len(individual_run_df), data.dtypes[y],
                                       outlier_temp_path)

                if cluster_tuple_lst:
                    records = [{"row_index": int(row), y: int(cid)} for row, cid in cluster_tuple_lst]
                    df = pd.DataFrame(records)
                    # Write to per-attribute directory
                    results_location = Path('.') / config_get('result_location')
                    clusters_temp_dir = results_location / "clusters_tmp"

                    # Make sure the directory exists
                    clusters_temp_dir.mkdir(parents=True, exist_ok=True)

                    map_dir = clusters_temp_dir / f"{y}"
                    map_dir.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(map_dir / f"part_{cur_partition_num:05d}.parquet", index=False)

            if (cur_partition_num + 1) % 10 == 0:
                gc.collect()

    log_memory_usage('After processing partitions')

    # -----------------------------
    # Postprocessing
    # -----------------------------
    postprocess_time = 0
    if cfg.postprocess_data:
        [clusters, new_num_outliers], postprocess_time = function_execution_in_milliseconds(
            free_clusters_to_best_compression, clusters, num_outliers, data.dtypes[y], len(data),
            Path('.') / config_get('result_location') / 'outliers.txt', y)
        num_outliers = new_num_outliers
        individual_run_df.at[individual_run_df.index[-1], 'num_clusters'] = len(clusters)

    # -----------------------------
    # Size stats
    # -----------------------------
    datatype_of_predicted = data.dtypes[y]
    size_stats = size(len(clusters), num_outliers, datatype_of_predicted, len(data), y, predicting_feature_count=len(x))

    individual_run_df.at[individual_run_df.index[-1], 'time_elapsed (ms)'] = (
            train_time + postprocess_time
    )
    individual_run_df.at[individual_run_df.index[-1], 'size (bytes)'] = size_stats[0]
    individual_run_df.at[individual_run_df.index[-1], 'original_size (bytes)'] = size_stats[1]
    individual_run_df.at[individual_run_df.index[-1], 'percentage_of_original_size'] = size_stats[2]
    individual_run_df.at[individual_run_df.index[-1], 'num_outliers'] = num_outliers
    individual_run_df.at[individual_run_df.index[-1], 'models_size (bytes)'] = size_stats[3]
    individual_run_df.at[individual_run_df.index[-1], 'outlier_size (bytes)'] = size_stats[4]
    individual_run_df.at[individual_run_df.index[-1], 'original_bits'] = size_stats[5]
    individual_run_df.at[individual_run_df.index[-1], 'bits'] = size_stats[6]

    # -----------------------------
    # Aggregate results
    # -----------------------------
    compressed_size = individual_run_df['size (bytes)'].astype(int).sum()
    original_size = individual_run_df['original_size (bytes)'].astype(int).sum()
    percentage = (compressed_size / original_size) * 100
    models_size = individual_run_df['models_size (bytes)'].astype(int).sum()
    outliers_size = individual_run_df['outlier_size (bytes)'].astype(int).sum()

    # -----------------------------
    # Error Results
    # -----------------------------
    if cfg.provide_error:
        outlier_ddf = dd.read_parquet(Path('.') / config_get('result_location') / 'errors' / f"{y}" / "outliers", engine="pyarrow")
        out_stats = outlier_ddf.describe(include="all").compute()
        out_median = outlier_ddf["outlier_errors"].quantile(0.5).compute()

        inlier_ddf = dd.read_parquet(Path('.') / config_get('result_location') / 'errors' / f"{y}" / "inliers", engine="pyarrow")
        in_stats = inlier_ddf.describe(include="all").compute()
        in_median = inlier_ddf["inlier_errors"].quantile(0.5).compute()

        print(f"{out_stats.loc['mean', 'outlier_errors']}")
    try:
        if compressed_size > original_size:
            raise ValueError(
                f'Compressed Size is larger than original size for {y}. Consider adjusting predictors or model.'
            )
    except ValueError:
        print(f'Compressed Size is larger than original size for {y}. Consider adjusting predictors or model.')

    new_row = {
        'predicting_feature': x,
        'predicted_feature': y,
        'ml_method': cfg.model_type,
        'clustering_method': individual_run_df.loc[len(individual_run_df) - 1, 'clustering_method'],
        'error_threshold': individual_run_df.loc[len(individual_run_df) - 1, 'error_threshold'],
        'min_split_size': individual_run_df.loc[len(individual_run_df) - 1, 'min_split_size'],
        'columns_decayed': len(individual_run_df),
        'num_outliers': individual_run_df['num_outliers'].astype(int).sum(),
        'size (bytes)': compressed_size,
        'original_size (bytes)': original_size,
        'percentage_of_original_size': f"{percentage:.4f}%",
        'time_elapsed (ms)': individual_run_df['time_elapsed (ms)'].astype(float).sum(),
        'batch_method': cfg.batch_method,
        'batch_size': cfg.batch_size,
        'models_size (bytes)': models_size,
        'outlier_size (bytes)': outliers_size,
        'original_bits': individual_run_df.loc[len(individual_run_df) - 1, 'original_bits'].astype(int),
        'bits': individual_run_df.loc[len(individual_run_df) - 1, 'bits'].astype(int),
        'mean_outlier_error': out_stats.loc['mean', 'outlier_errors'] if cfg.provide_error else None,
        'max_outlier_error': out_stats.loc['max', 'outlier_errors'] if cfg.provide_error else None,
        'min_outlier_error': out_stats.loc['min', 'outlier_errors'] if cfg.provide_error else None,
        'median_outlier_error': out_median if cfg.provide_error else None,
        'std_outlier_error': out_stats.loc['std', 'outlier_errors'] if cfg.provide_error else None,
        'mean_inlier_error': in_stats.loc['mean', 'inlier_errors'] if cfg.provide_error else None,
        'max_inlier_error': in_stats.loc['max', 'inlier_errors'] if cfg.provide_error else None,
        'min_inlier_error': in_stats.loc['min', 'inlier_errors'] if cfg.provide_error else None,
        'median_inlier_error': in_median if cfg.provide_error else None,
        'std_inlier_error': in_stats.loc['std', 'inlier_errors'] if cfg.provide_error else None,
    }

    overall_results_df.loc[len(overall_results_df)] = new_row

    if cfg.persist_structures:
        results_location = Path('.') / config_get('result_location')
        cluster_model_path = results_location / 'models'
        create_file_if_not_exists(cluster_model_path)
        persist_cluster(clusters, cluster_model_path, cfg.model_type, x, y, cfg.batch_size)

    # -----------------------------
    # Cleanup
    # -----------------------------
    for var in ['x_index', 'y_index', 'kd_index', 'clusters', 'holder']:
        if var in locals():
            del locals()[var]

    # clear_temp_dir(Path('.') / config_get('result_location') / 'errors' / f"{y}" / "inliers")
    # clear_temp_dir(Path('.') / config_get('result_location') / 'errors' / f"{y}" / "outliers")

    # Clear TF/Keras if needed
    try:
        tf.keras.backend.clear_session()
    except AttributeError as e:
        print(f"Warning: Could not clear Tensorflow session: {e}")
    except RuntimeError as e:
        print(f"Warning: Tensorflow session error: {e}")
    finally:
        gc.collect()

    return individual_run_df, overall_results_df


def determine_clusters(cfg: RuntimeConfig, individual_run_df: pd.DataFrame, cur_data, x, y, x_list, column_info_table):
    """
    Determine clusters and return (results, train_time).
    """

    # Multivariable branch
    if cfg.model_type == 'multivariable LR':
        raise ValueError("Multivariable LR not available in this version.")
    # Non-multivariable branch
    else:
        # Text model case
        if x_list[0] != 'all_zeroes' and column_info_table[x][0] == 'string' and column_info_table[y][0] == 'string':
            raise ValueError("Textual Data not available in this version.")
        # Supervised clustering
        elif cfg.clustering == "supervised":
            return function_execution_in_milliseconds(train_model, individual_run_df, cur_data, x, y, cfg.cluster_alg,
                                                      cfg.outlier_before, cfg.outlier_after, cfg.accuracy_tuning,
                                                      cfg.accuracy, cfg.planned_clusters, model_type=cfg.model_type)
        # Unsupervised clustering
        elif cfg.clustering == "unsupervised":
            return function_execution_in_milliseconds(train_model_unsupervised, individual_run_df, cur_data, x, y,
                                                      cfg.cluster_alg, cfg.accuracy, cfg.split_size,
                                                      cfg.preprocess_data, model_type=cfg.model_type)
        # No clustering / no outlier detection
        else:
            return function_execution_in_milliseconds(train_no_cluster_outliers, individual_run_df, cur_data, x, y,
                                                      model_type=cfg.model_type)


def process_partition(data, cur_partition_num: int, cfg: RuntimeConfig, x: list[str] | str, y: str,
                      column_index_variable: str, x_index, y_index, kd_index, clusters,
                      inlier_cluster_location, missing_value_methods, x_replacement_vals
                      ):
    """
    Process a single partition using specified indexes and returns (holder, fitting_time, num_outliers).
    """

    # Select columns depending on model type
    if cfg.model_type == 'multivariable LR':
        raise ValueError("Multivariable LR not available in this version.")

    else:
        cur_batch = data.get_partition(cur_partition_num)[[column_index_variable, x, y]].compute()

        holder, fitting_time = function_execution_in_milliseconds(
            fit_new_batch,
            cur_batch,
            cfg.batch_method,
            clusters,
            x,
            y,
            x_index,
            y_index,
            cfg.accuracy,
            clustering_destination=inlier_cluster_location,
            cur_partition_num=cur_partition_num,
            model_type=cfg.model_type,
            null_method=missing_value_methods[y]['method'],
            replacement_x=x_replacement_vals,
            replacement_y=missing_value_methods[y]['replacement_value'],
            provide_error=cfg.provide_error
        )

    holder_outliers = holder[0]
    holder_clustered = holder[1]
    holder_outlier_errors = holder[2]
    holder_inlier_errors = holder[3]
    num_outliers = len(holder_outliers)
    outlier_values = (cur_batch[cur_batch[column_index_variable].isin(holder_outliers)])[y].tolist()

    # Cleanup
    del cur_batch

    return outlier_values, holder_outliers, fitting_time, num_outliers, holder_clustered, holder_outlier_errors, holder_inlier_errors


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


def single_predictor_batch_sample(data, sample_max_attempts, column_index_variable, x, y, missing_value_methods):
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
        data_to_sample = data[[column_index_variable, x, y]]
        sampled_data = data_to_sample.sample(frac=approx_sample_percentage_per_partition,
                                             random_state=(sampling_seed + cur_sample_attempt))
        cur_data = sampled_data.compute()

        missing_value_method_x = missing_value_methods[x]['method']
        missing_value_method_y = missing_value_methods[y]['method']

        # Both x and y must agree on their methods of replacement. Otherwise, it's unclear which behavior should be used
        if missing_value_method_x != missing_value_method_y:
            print(
                f"The null replacement method of: {x} and {y} must match! Please correct the config and try again...")
            sys.exit(0)

        # Both must match at this point, only need to check one
        if missing_value_method_y == 'outliers':
            print(f'Dropping nulls found in: [{x}, {y}]')
            cur_data = cur_data.dropna(axis=0, subset=[x, y])
        elif missing_value_method_y == 'replace_missing_value':
            x_replacement_val = missing_value_methods[x]['replacement_value']
            y_replacement_val = missing_value_methods[y]['replacement_value']
            print(
                f'Replacing nulls found in: [{x}, {y}] with special values: {x_replacement_val} and {y_replacement_val}')
            cur_data = cur_data.fillna(value={x: x_replacement_val, y: y_replacement_val})
        else:
            print(f'Unknown replacement method for {x} or {y}. Please correct the config and try again...')
            sys.exit(0)

        if not cur_data.empty:
            valid_sample = True

            cur_data = cur_data.reset_index(drop=True)
            print('Finished sampling, proceeding to calculate clusters')
        else:
            print('Empty sample due to Null Values, attempting to resample.')

        cur_sample_attempt += 1
    if cur_data.empty:
        print(
            f'Unable to sample a non-empty dataset after {sample_max_attempts} attempts. Exiting program.')
        print(
            'Consider trying a different seed, increasing the number of attempts, and analyzing the dataset.')
        sys.exit(0)

    return cur_data


def multi_predictor_batch_sample(data, sample_max_attempts, column_index_variable, x, y, missing_value_methods):
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
    projection_lst = x + [column_index_variable, y]

    # Iteration allows retrials of sampling in the event that no rows are sampled or all have Nulls
    while not valid_sample and cur_sample_attempt < sample_max_attempts:
        data_to_sample = data[projection_lst]
        sampled_data = data_to_sample.sample(frac=approx_sample_percentage_per_partition,
                                             random_state=(sampling_seed + cur_sample_attempt))
        cur_data = sampled_data.compute()

        missing_value_methods_lst = [missing_value_methods[predictor]['method'] for predictor in x]
        missing_value_methods_lst.append(missing_value_methods[y]['method'])

        # All predictors and y must agree on their methods of replacement. Otherwise, it's unclear which behavior should be used
        if len(set(missing_value_methods_lst)) != 1:
            print(
                f"The null replacement methods of: {x} and {y} must match! Please correct the config and try again...")
            sys.exit(0)

        # Both must match at this point, only need to check one
        if missing_value_methods_lst[0] == 'outliers':
            print(f'Dropping nulls found in: [{x}, {y}]')
            null_dropped_col_lst = x + [y]
            cur_data = cur_data.dropna(axis=0, subset=null_dropped_col_lst)
        elif missing_value_methods_lst[0] == 'replace_missing_value':
            replacement_dict = {}
            for predictor in x:
                replacement_dict[predictor] = missing_value_methods[predictor]['replacement_value']
            replacement_dict[y] = missing_value_methods[y]['replacement_value']
            print(
                f'Replacing nulls found in: [{x}, {y}] with special values: {replacement_dict}')
            cur_data = cur_data.fillna(value=replacement_dict)
        else:
            print(f'Unknown replacement method for {x} or {y}. Please correct the config and try again...')
            sys.exit(0)

        if not cur_data.empty:
            valid_sample = True

            cur_data = cur_data.reset_index(drop=True)
            print('Finished sampling, proceeding to calculate clusters')
        else:
            print('Empty sample due to Null Values, attempting to resample.')

        cur_sample_attempt += 1
    if cur_data.empty:
        print(
            f'Unable to sample a non-empty dataset after {sample_max_attempts} attempts. Exiting program.')
        print(
            'Consider trying a different seed, increasing the number of attempts, and analyzing the dataset.')
        sys.exit(0)

    return cur_data


if __name__ == "__main__":
    _, millis = function_execution_in_milliseconds(main)
    print(f"Total time: {format_duration(millis)}")
