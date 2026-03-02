import bz2
import gzip
import lzma
import pickle
import shutil
import snappy
import sys
import zstandard
from pysz import sz, szConfig, szErrorBoundMode, szAlgorithm
import zfpy
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import statistics

from config import config_get
from helper import function_execution_in_milliseconds, list_of_percent_differences

file_to_compress = config_get('database')
compression_params = config_get('compression_parameters')

tolerance = 1e-2


def compression_wrapper(data: pd.DataFrame, method: str, compressed_columns: list[str], write_to_file=True) -> tuple[
    float, float, float, float, float]:

    original_size = sys.getsizeof(df_to_bytes(data))

    if method == 'sz' or method == 'zfp':
        error_rslts_table = pd.DataFrame(
            columns=['method', 'column', 'outliers', 'inliers', 'tolerance', 'global_mean_error', 'global_median_error',
                     'global_min_error', 'global_max_error', 'global_stdev_error', 'inliers_mean_error',
                     'inliers_median_error', 'inliers_min_error', 'inliers_max_error', 'inliers_stdev_error',
                     'outliers_mean_error', 'outliers_median_error', 'outliers_min_error', 'outliers_max_error',
                     'outliers_stdev_error'])
        compressed_size, decompressed_size = 0, 0
        compression_time, decompression_time = 0, 0
        for cur_col in tqdm(compressed_columns, desc='Compressing Columns', colour='green', leave=False):
            print(f'Results of {cur_col}:')
            original_data = data[cur_col].to_numpy()
            error_rslt = pd.DataFrame()

            compressed_data, cur_compression_time = function_execution_in_milliseconds(_compress_numpy, original_data,
                                                                                       method)
            decompressed_data, cur_decompression_time = function_execution_in_milliseconds(_decompress_numpy,
                                                                                           compressed_data, method,
                                                                                           data_shape=original_data.shape)

            if method == 'sz':
                compressed_size += compressed_data.size
            else:
                compressed_size += sys.getsizeof(compressed_data)
            decompressed_size += sys.getsizeof(decompressed_data)
            compression_time += cur_compression_time
            decompression_time += cur_decompression_time

            error_rslt['abs_percentage_error'] = list_of_percent_differences(original_data.tolist(),
                                                                             decompressed_data.tolist())
            inliers_table = error_rslt[error_rslt["abs_percentage_error"] <= (tolerance * 100)]
            outliers_table = error_rslt[error_rslt["abs_percentage_error"] > (tolerance * 100)]
            global_mean_error = np.mean(error_rslt['abs_percentage_error'])
            global_median_error = np.median(error_rslt['abs_percentage_error'])
            global_min_error = np.min(error_rslt['abs_percentage_error'])
            global_max_error = np.max(error_rslt['abs_percentage_error'])
            global_stdev_error = np.std(error_rslt['abs_percentage_error'])

            inliers_mean_error = np.mean(inliers_table['abs_percentage_error'])
            inliers_median_error = np.median(inliers_table['abs_percentage_error'])
            inliers_min_error = np.min(inliers_table['abs_percentage_error'])
            inliers_max_error = np.max(inliers_table['abs_percentage_error'])
            inliers_stdev_error = np.std(inliers_table['abs_percentage_error'])

            outliers_mean_error = np.mean(outliers_table['abs_percentage_error'])
            outliers_median_error = np.median(outliers_table['abs_percentage_error'])
            outliers_min_error = np.min(outliers_table['abs_percentage_error'])
            outliers_max_error = np.max(outliers_table['abs_percentage_error'])
            outliers_stdev_error = np.std(outliers_table['abs_percentage_error'])

            error_rslts_table.loc[len(error_rslts_table)] = {'method': method, 'column': cur_col,
                                                             'outliers': len(outliers_table),
                                                             'inliers': len(inliers_table),
                                                             'tolerance': f"{tolerance * 100}%",
                                                             'global_mean_error': global_mean_error,
                                                             'global_median_error': global_median_error,
                                                             'global_min_error': global_min_error,
                                                             'global_max_error': global_max_error,
                                                             'global_stdev_error': global_stdev_error,
                                                             'inliers_mean_error': inliers_mean_error,
                                                             'inliers_median_error': inliers_median_error,
                                                             'inliers_min_error': inliers_min_error,
                                                             'inliers_max_error': inliers_max_error,
                                                             'inliers_stdev_error': inliers_stdev_error,
                                                             'outliers_mean_error': outliers_mean_error,
                                                             'outliers_median_error': outliers_median_error,
                                                             'outliers_min_error': outliers_min_error,
                                                             'outliers_max_error': outliers_max_error,
                                                             'outliers_stdev_error': outliers_stdev_error}
        print(error_rslts_table)
        if write_to_file:
            results_location = Path('.') / 'results' / 'compression' / 'error_results.csv'

            if not os.path.exists(results_location):
                os.makedirs(os.path.dirname(results_location), exist_ok=True)

            if os.path.exists(results_location):
                error_rslts_table.to_csv(results_location, mode='a', header=True, index=False)
            else:
                error_rslts_table.to_csv(results_location, mode='w', header=True, index=False)
    else:
        original_data = df_to_bytes(data[compressed_columns])

        compressed_data, compression_time = function_execution_in_milliseconds(_compress_data, original_data, method)
        decompressed_data, decompression_time = function_execution_in_milliseconds(_decompress_data, compressed_data,
                                                                                   method)
        compressed_size = sys.getsizeof(compressed_data)
        decompressed_size = sys.getsizeof(decompressed_data)

    # Add cost of excluded columns
    excl_columns = [item for item in data.columns if item not in compressed_columns]
    excl_size = sys.getsizeof(df_to_bytes(data[excl_columns]))
    compressed_size += excl_size
    return compression_time, decompression_time, original_size, compressed_size, decompressed_size


def _compress_numpy(data: np.ndarray, method: str):
    match method:
        case 'sz':
            config = szConfig()
            config.errorBoundMode = szErrorBoundMode.REL
            config.relErrorBound = tolerance
            # config.errorBoundMode = szErrorBoundMode.ABS
            # config.absErrorBound = tolerance
            config.cmprAlgo = szAlgorithm.LORENZO_REG  # Lossless only
            compressed, ratio = sz.compress(data, config)
            print(f'Compressed size to {compressed.size} bytes ({ratio:.2f}x) ')
            return compressed
        case 'zfp':
            compressed = zfpy.compress_numpy(data, tolerance=tolerance)
            return compressed


def _decompress_numpy(compressed_data, method: str, data_shape=None):
    match method:
        case 'sz':
            decompressed, dec_config = sz.decompress(compressed_data, np.float64, data_shape)
            return decompressed
        case 'zfp':
            return zfpy.decompress_numpy(compressed_data)


def _compress_data(uncompressed_data: bytes, method: str) -> bytes:
    match method:
        case 'lzma':
            return lzma.compress(uncompressed_data)
        case 'gzip':
            return gzip.compress(uncompressed_data)
        case 'bzip2':
            return bz2.compress(uncompressed_data)
        case 'zstd':
            return zstandard.compress(uncompressed_data)
        case 'snappy':
            return snappy.compress(uncompressed_data)
        case _:
            raise ValueError("Please select a valid compression algorithm and try again. [lzma, lzw]")


def _compress(uncompressed_path: str, compressed_path: str):
    with open(uncompressed_path, "rb") as uncompressed_file:
        match compression_params["method"]:
            case 'lzma':
                with lzma.open(compressed_path, "wb") as compressed_file:
                    shutil.copyfileobj(uncompressed_file, compressed_file)
            case 'gzip':
                with gzip.open(compressed_path, "wb") as compressed_file:
                    shutil.copyfileobj(uncompressed_file, compressed_file)
            case 'bzip2':
                with bz2.open(compressed_path, "wb") as compressed_file:
                    shutil.copyfileobj(uncompressed_file, compressed_file)
            case 'zstd':
                with zstandard.open(compressed_path, "wb") as compressed_file:
                    shutil.copyfileobj(uncompressed_file, compressed_file)
            case 'snappy':
                with open(compressed_path, "wb") as compressed_file:
                    snappy.stream_compress(uncompressed_file, compressed_file)
            case _:
                raise ValueError("Please select a valid compression algorithm and try again. [lzma, lzw]")


def _decompress_data(compressed_data: bytes, method: str) -> bytes:
    match method:
        case 'lzma':
            return lzma.decompress(compressed_data)
        case 'gzip':
            return gzip.decompress(compressed_data)
        case 'bzip2':
            return bz2.decompress(compressed_data)
        case 'zstd':
            return zstandard.decompress(compressed_data)
        case 'snappy':
            return snappy.decompress(compressed_data)
        case _:
            raise ValueError("Please select a valid compression algorithm and try again. [lzma, lzw]")


def _decompress(compressed_path: str, expanded_path: str):
    match compression_params["method"]:
        case 'lzma':
            with lzma.open(compressed_path, "rb") as compressed_file:
                with open(expanded_path, "wb") as uncompressed_file:
                    shutil.copyfileobj(compressed_file, uncompressed_file)
        case 'gzip':
            with gzip.open(compressed_path, "rb") as compressed_file:
                with open(expanded_path, "wb") as uncompressed_file:
                    shutil.copyfileobj(compressed_file, uncompressed_file)
        case 'bzip2':
            with bz2.open(compressed_path, "rb") as compressed_file:
                with open(expanded_path, "wb") as uncompressed_file:
                    shutil.copyfileobj(compressed_file, uncompressed_file)
        case 'zstd':
            with zstandard.open(compressed_path, "rb") as compressed_file:
                with open(expanded_path, "wb") as uncompressed_file:
                    shutil.copyfileobj(compressed_file, uncompressed_file)
        case 'snappy':
            with open(compressed_path, "rb") as compressed_file:
                with open(expanded_path, "wb") as uncompressed_file:
                    snappy.stream_decompress(compressed_file, uncompressed_file)
        case _:
            raise ValueError("Please select a valid compression algorithm and try again. [lzma, lzw]")


def df_to_bytes(df: pd.DataFrame) -> bytes:
    return pickle.dumps(df)


def bytes_to_df(obj):
    return pickle.load(obj)


def main():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)

    if len(sys.argv) != 2:
        print("Please include a path to desired YAML file as follows: python3 compression.py <path_to_yaml>")
        sys.exit(1)

    runtime_parameters = config_get('runtime_parameters')

    columns = [
        'compression_method', 'columns_decayed', 'size (bytes)', 'original_size (bytes)',
        'percentage_of_original_size', 'time_elapsed (ms)', 'compression_time (ms)', 'decompression_time (ms)'
    ]
    output_summary = pd.DataFrame(columns=columns)
    print('Reading Data...')
    data = pd.read_csv(config_get('database'), sep=',')

    for choice, parameters in runtime_parameters.items():
        compression_params = parameters.get('compression_parameters')
        method = compression_params.get('method')
        compressed_columns = compression_params.get('compressed_columns')
        encoding = compression_params.get('encoding')

        print(f'Running compression with {method}')

        results, total_time = function_execution_in_milliseconds(compression_wrapper, data, method, compressed_columns)
        compression_time = results[0]
        decompression_time = results[1]
        original_size = results[2]
        compressed_size = results[3]
        decompressed_size = results[4]

        new_row = {
            'compression_method': method,
            'columns_decayed': len(compressed_columns),
            'size (bytes)': compressed_size,
            'original_size (bytes)': original_size,
            'percentage_of_original_size': str(round(compressed_size / original_size * 100, 4)) + '%',
            'time_elapsed (ms)': total_time,
            'compression_time (ms)': compression_time,
            'decompression_time (ms)': decompression_time,
        }
        output_summary.loc[len(output_summary)] = new_row

    print(output_summary)


if __name__ == "__main__":
    main()
