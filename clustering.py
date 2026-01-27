import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, BisectingKMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN


#################################################### Clustering

def k_means(data_file: pd.DataFrame, numClusters: int, x_label: str, y_label: str, binary=False) -> list:
    x = data_file[x_label]
    y = data_file[y_label]

    data = []
    if binary:
        for i in range(len(x)):
            data.append(x[i] + y[i])
    else:
        data = list(zip(x, y))

    try:
        x_sub = x[0]
        y_sub = y[0]
    except KeyError:
        x_sub = None
        y_sub = None

    if isinstance(x, pd.core.series.Series) and isinstance(x_sub, np.ndarray) and isinstance(y,
                                                                                             pd.core.series.Series) and isinstance(
            y_sub, np.ndarray):
        adjusted_data = []
        for i in range(len(x)):
            adjusted_data.append(list(x[i]) + list(y[i]))
        data = adjusted_data

    # actually run algorithm
    kmeans = KMeans(n_clusters=numClusters)
    kmeans.fit(data)

    # Create a DataFrame with the original data and cluster labels
    copied_data = data_file.copy()
    copied_data['Cluster'] = kmeans.labels_

    # Create a list of clusters
    cluster_list = []
    # Create a new DataFrame for each cluster
    for i in range(numClusters):
        # create a scatter plot for each cluster
        cluster = copied_data[copied_data['Cluster'] == i]
        cluster_list.append(cluster)

    return cluster_list


# Method for distribution clustering
def distribution(data: pd.DataFrame, num_clusters: int, x_label: str, y_label: str) -> list:
    # To use GMM for clustering, you need to prepare the data by selecting the relevant columns and converting them into a numpy array.
    X = data[['Index', x_label, y_label]].values

    gmm = GaussianMixture(n_components=num_clusters)  # number of clusters
    gmm.fit(X)

    labels = gmm.predict(X)

    fig, axs = plt.subplots(1, num_clusters, figsize=(5 * num_clusters, 5))
    clusters = []
    for i in range(num_clusters):
        cluster = X[labels == i]
        axs[i].scatter(cluster[:, 0], cluster[:, 1], c='b')
        axs[i].set_xlabel(x_label)
        axs[i].set_ylabel(y_label)
        axs[i].set_title(f'Cluster {i + 1}')

        # convert array back to a dataframe
        dataset = pd.DataFrame({'Index': cluster[:, 0], x_label: cluster[:, 1], y_label: cluster[:, 2]})
        clusters.append(dataset)

    return clusters


# DB scan is inefficient and works
# Better for outlier detection, not clustering
def db_scan(data, input, x_label, y_label):
    nbrs = NearestNeighbors(n_neighbors=5).fit(data)

    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(data)

    sort_neigh_dist = np.sort(neigh_dist, axis=0)

    k_dist = sort_neigh_dist[:, 4]

    kneedle = KneeLocator(x=range(1, len(neigh_dist) + 1), y=k_dist, S=1.0,
                          curve="concave", direction="increasing", online=True)
    clusters = DBSCAN(eps=input, min_samples=4).fit(data)
    data['Cluster'] = clusters.labels_

    cluster_list = []
    for i in np.unique(clusters.labels_):
        cluster_list.append(data[data['Cluster'] == i])

    return cluster_list


def minhal_split(data_file: pd.DataFrame, model, x_label: str, y_label: str, binary=False) -> list:
    """
    Basic regression-based clustering that splits data into 2 clusters 
    based on points above/below regression line.
    """
    x = data_file[x_label].values.reshape(-1, 1)
    y = data_file[y_label].values

    predicted_y = model.predict(x)

    # Split into 2 clusters: points above and below the line
    cluster1 = data_file[y > predicted_y].copy()
    cluster2 = data_file[y <= predicted_y].copy()

    return [cluster1, cluster2]


def birch(data_file: pd.DataFrame, numClusters: int, x_label: str, y_label: str, binary=False) -> list:
    x = data_file[x_label]
    y = data_file[y_label]
    data = np.column_stack((x, y))

    birch_model = Birch(n_clusters=numClusters)
    birch_model.fit(data)

    copied_data = data_file.copy()
    copied_data['Cluster'] = birch_model.labels_

    cluster_list = []
    for i in range(numClusters):
        cluster = copied_data[copied_data['Cluster'] == i]
        cluster_list.append(cluster)

    return cluster_list


def bisecting_kmeans(data_file: pd.DataFrame, numClusters: int, x_label: str, y_label: str, binary=False) -> list:
    x = data_file[x_label]
    y = data_file[y_label]
    data = np.column_stack((x, y))

    bisecting_kmeans_model = BisectingKMeans(n_clusters=numClusters, init='k-means++',
                                             bisecting_strategy='largest_cluster')
    bisecting_kmeans_model.fit(data)

    copied_data = data_file.copy()
    copied_data['Cluster'] = bisecting_kmeans_model.labels_

    cluster_list = []
    for i in range(numClusters):
        cluster = copied_data[copied_data['Cluster'] == i]
        cluster_list.append(cluster)

    return cluster_list
