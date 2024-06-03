import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from analysis.analyse_dataset import find_most_correlated
from models.utils import find_best
from plot.plot import plot_with_text
from format_data.process_data import process_data


def k_means(input_file, random_state, sample_size, silhouette_score_file, save_k_means_file):
    df = pd.read_csv(input_file)
    # Replace null values
    preprocessed_data = process_data('most_frequent', df)
    # Sample so that silhouette_score performing is faster
    sample_data = create_sample_data(random_state, preprocessed_data, sample_size)
    n_clusters = find_best_n_clusters(sample_data, random_state, silhouette_score_file)
    plot_clusters(preprocessed_data, n_clusters, random_state, save_k_means_file)


def create_sample_data(random_state, data, sample_size):
    np.random.seed(random_state)
    indices = np.random.choice(data.shape[0], sample_size, replace=False)
    data_sample = data[indices]
    return data_sample


def kmeans_score(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    return silhouette_score(data, cluster_labels)


def find_best_n_clusters(data, random_state, save_plot_file):
    x = range(2, 31)
    max_silhouette_score, best_n_clusters, silhouette_score_avg = find_best(data, x, kmeans_score, 2)

    title = 'Silhouette score for number of cluster from 2 to 20'
    x_label = 'Number of cluster'
    y_label = 'Silhouette score'
    plot_with_text(x, silhouette_score_avg, title, x_label, y_label, best_n_clusters, max_silhouette_score,
                   save_plot_file)
    return best_n_clusters


def plot_clusters(data, n_clusters, random_state, save_file=None):
    pca = PCA(n_components=2)
    pca_transformed_data = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(data)

    # Plot cluster according to PCA
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_transformed_data[:, 0], pca_transformed_data[:, 1], c=cluster_labels, cmap='viridis',
                          s=50)
    plt.title(f'KMeans Clustering Results with PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(scatter, label='Cluster')
    if save_file is not None:
        plt.savefig(save_file)
