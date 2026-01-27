import pandas as pd

from tqdm import tqdm
from config import all_column_info
from train_model import train_no_cluster_outliers, train_model, train_model_unsupervised

"""
Class creates a table where there all valid features are inputs for all valid features. Additionally, things like the 
Index or the all_zeroes column can be selected as input. If the user specifies the input they want, this class will also create 
the dictionary for pipeline.py to use.
"""


class Feature:
    def __init__(self, name: str, type, size: int):
        self.name = name
        self.type = type
        self.size = int(size)

    def __str__(self):
        return f"Name: {self.name}, Type: {self.type}, Size: {self.size}"


class TableEntry:
    def __init__(self, x_feature: Feature, y_feature: Feature, compression_rate=0):
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.compression_rate = compression_rate

    def update_compression_rate(self, compression_rate: int):
        self.compression_rate = compression_rate

    def __str__(self):
        return self.x_feature.name + "->" + self.y_feature.name + "=" + str(self.compression_rate)


class FeatureTable:
    def __init__(self, x_features: list[Feature], y_features: list[Feature]):
        self.table = [[TableEntry(x_feature, y_feature, 0) for y_feature in y_features] for x_feature in x_features]
        self.feature_indices_x = {feature: i for i, feature in enumerate(x_features)}
        self.feature_indices_y = {feature: i for i, feature in enumerate(y_features)}

        for i, x_feature in enumerate(x_features):
            for j, y_feature in enumerate(y_features):
                self.table[i][j] = TableEntry(x_feature, y_feature)

    def update_value(self, feature1: Feature, feature2: Feature, value: float):
        i = self.feature_indices_x.get(feature1)
        j = self.feature_indices_y.get(feature2)
        if i is not None and j is not None:
            self.table[i][j].compression_rate = value
        else:
            raise ValueError("One or more features not found")

    def get_value(self, feature1: Feature, feature2: Feature) -> int:
        i = self.feature_indices_x.get(feature1)
        j = self.feature_indices_y.get(feature2)
        if i is not None and j is not None:
            return self.table[i][j].compression_rate
        else:
            raise ValueError("One or more features not found")

    def get_feature_with_lowest_sum(self, num_rows_in_dataset: int) -> Feature | None:
        min_sum = float('inf')
        x_with_lowest_y_sum = None
        for i, x_feature in enumerate(self.table):
            sum_y = sum(entry.compression_rate for entry in x_feature)
            if x_feature[0].x_feature != 'all_zeroes':
                sum_y += x_feature[0].x_feature.size * num_rows_in_dataset
            if sum_y < min_sum:
                min_sum = sum_y
                x_with_lowest_y_sum = x_feature[0].x_feature  # Assuming all entries in the row have the same x_feature
        return x_with_lowest_y_sum

    def print_table(self):
        print("compression table")
        for row in self.table:
            for entry in row:
                print(entry, end="\t")
            print()


def select_best_features(data: pd.DataFrame, clustering: str, cluster_alg: str, accuracy: float, split_size: int,
                         outlier_before: bool, outlier_after: bool,
                         accuracy_tuning: bool, planned_clusters: int, preprocess_data: bool, postprocess_data: bool,
                         random_sample_size: float) -> dict:
    run_results = pd.DataFrame(
        columns=['clustering_method', 'accuracy_threshold', 'min_split_size', 'num_clusters', "num_outliers",
                 'size (bytes)',
                 'original_size (bytes)', 'percentage_of_original_size', 'average_percent_difference', 'mse',
                 'time_elapsed (ms)', 'recovered_accuracy'])

    valid_features_y = remove_features_below_size(load_features())

    all_zero_feature = Feature('all_zeroes', int, 64)
    valid_features_x = valid_features_y.copy()
    valid_features_x.append(all_zero_feature)

    compression_table = FeatureTable(valid_features_x, valid_features_y)
    compression_table.print_table()

    sampled_data = data.sample(frac=random_sample_size)

    feature_pairs = []
    for x in valid_features_x:
        for y in valid_features_y:
            if x != y:
                feature_pairs.append((x, y))

    for pair in tqdm(feature_pairs, desc="Testing all inputs/outputs", ascii=' ='):
        process_feature_pair(clustering, run_results, sampled_data, cluster_alg, outlier_before, outlier_after,
                             accuracy_tuning, accuracy, planned_clusters, split_size, preprocess_data, postprocess_data,
                             pair[0], pair[1], compression_table)

    best_predictor = compression_table.get_feature_with_lowest_sum(len(data))
    print("Best Predictor: " + best_predictor.name)
    xy_pairs = create_dictionary_based_on_best(best_predictor, valid_features_y)
    return xy_pairs


def process_feature_pair(clustering: str, run_results: pd.DataFrame, sampled_data: pd.DataFrame, cluster_alg: str,
                         outlier_before: bool, outlier_after: bool,
                         accuracy_tuning: bool, accuracy: float, planned_clusters: int, split_size: int,
                         preprocess_data: bool, postprocess_data: bool, x: Feature,
                         y: Feature, compression_table: FeatureTable):
    if clustering == 'None':
        results, time = train_no_cluster_outliers(run_results, sampled_data, x.name, y.name)
    elif clustering == "supervised":
        results, time = train_model(run_results, sampled_data, x.name, y.name, cluster_alg, outlier_before,
                                    outlier_after, accuracy_tuning, accuracy, planned_clusters)
    elif clustering == "unsupervised":
        results, time = train_model_unsupervised(run_results, sampled_data, x.name, y.name, cluster_alg, accuracy,
                                                 split_size, preprocess_data, postprocess_data)

    last_row = run_results.iloc[-1]
    percentage_value = float(last_row['percentage_of_original_size'].strip('%'))

    compression_table.update_value(x, y, percentage_value)
    compression_table.print_table()


def create_dictionary_based_on_best(best_predictor: Feature, valid_features: list) -> dict:
    xy_pairs = {}
    for feature in valid_features:
        if best_predictor.name != feature.name:
            xy_pairs[feature.name] = [best_predictor.name]

    return xy_pairs


def remove_features_below_size(features: list[Feature], minimum_size=4) -> list[Feature]:
    filtered_features = [feature for feature in features if
                         feature.size >= minimum_size and feature.type != 'index' and feature.type != 'date']
    return filtered_features


def load_features() -> list[Feature]:
    feature_list = []
    features = all_column_info()
    for key, tuple in features.items():
        current_feature = Feature(key, tuple[0], tuple[1])
        feature_list.append(current_feature)

    return feature_list


def create_dictionary_with_predictor(predictor: str) -> dict[str, list[str]]:
    valid_features = remove_features_below_size(load_features())
    best_feature = Feature(predictor, None, 0)
    xy_pairs = create_dictionary_based_on_best(best_feature, valid_features)
    return xy_pairs

