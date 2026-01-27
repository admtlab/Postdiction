import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from config import config_get, column_variable_get


def outlier_detection(data: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config_get("outlier_detection_technique") == "z_score":
        return zScore(data, threshold)
    elif config_get("outlier_detection_technique") == "i_forest":
        return iForest(data, threshold)
    elif config_get("outlier_detection_technique") == "combo":
        return combo(data, threshold)
    else:
        raise ValueError(
            f"Invalid config value asssociated with key outlier_detection_technique: {config_get('outlier_detection_technique')}. Values can only be z_score or i_forest.")


# Uses the z score outlier detection method. is an outlier if eigther column is an outlier
def zScore(data: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_copy = remove_unwanted_datatypes(data)
    # Calculate z-scores for each variable
    z_scores = (data_copy - data_copy.mean()) / data_copy.std()
    # Calculate absolute z-scores
    abs_z_scores = np.abs(z_scores)
    # Find and mark outliers
    outliers = abs_z_scores > threshold
    # Remove outliers so this can be further edited  
    no_outliers = data_copy[~outliers.any(axis=1)]
    # Extract the rows with outliers
    outliers_only = data_copy[outliers.any(axis=1)]

    return outliers_only, no_outliers


# uses the iForest technique of outlier detection
def iForest(data: pd.DataFrame, ignore: float, n_estimators=100,
            outlier_fraction='auto') -> tuple[
    pd.DataFrame, pd.DataFrame]:  # Outlier fraction represents the proportion of data points that are expected to be outliers in the dataset
    data_copy = remove_unwanted_datatypes(data)

    # .01 means 1 percent outliers

    # Create an Isolation Forest object
    # n_estimators represents the number of decision trees that will be used to build the forest. higher the number the better peformance 50-500 good range
    isolation_forest = IsolationForest(n_estimators=n_estimators, contamination=outlier_fraction)
    # Fit the model to the data
    isolation_forest.fit(data_copy)
    # Use the model to predict the outliers
    outliers = isolation_forest.predict(data_copy) == -1
    # Extract the rows with outliers
    outliers_only = data_copy[outliers]
    # Extract the rows without outliers
    no_outliers = data_copy[~outliers]
    return outliers_only, no_outliers


def combo(data: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    zScore_outliers_only, zScore_no_outliers = zScore(data, threshold)
    iForest_outliers_only, iForest_no_outliers = iForest(data, threshold)

    outliers_only = pd.merge(zScore_outliers_only, iForest_outliers_only, how='inner',
                             on=list(zScore_outliers_only.columns))
    no_outliers = pd.merge(zScore_no_outliers, iForest_no_outliers, how='inner', on=list(zScore_no_outliers.columns))

    return outliers_only, no_outliers


#################################################### Helper methods
# for plotting the outliers on a graph
def plot_outliers(data: pd.DataFrame, outliers_only: pd.DataFrame, no_outliers: pd.DataFrame, title: str, x_label: str,
                  y_label: str):
    # Plot the data with outliers in red and data without outliers in blue
    plt.scatter(no_outliers[x_label], no_outliers[y_label], color='blue', label='Non-Outliers')
    plt.scatter(outliers_only[x_label], outliers_only[y_label], color='red', label='Outliers')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def remove_unwanted_datatypes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove datatypes that cannot be included in the clustering method. Currently, the only such datatype is `date`
    """
    data_copy = data.copy()
    for column_name in data_copy.columns:

        column_type = column_variable_get(column_name)

        if column_type[0] == 'date' or column_type[0] == 'datetime':
            data_copy = data_copy.drop(columns=[column_name])

    return data_copy
