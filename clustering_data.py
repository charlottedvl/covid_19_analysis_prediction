import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from plot.plot import plot_with_text


def preprocess_data(data):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    return data_scaled


def find_best_k(data, random_state, save_plot_file):
    silhouette_score_avg = []
    for n_clusters in range(2, 21):
        print(n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_score_avg.append(silhouette_avg)

    max_silhouette_score = max(silhouette_score_avg)
    best_n_clusters = silhouette_score_avg.index(max_silhouette_score) + 2

    x = range(2, 21)
    title = 'Silhouette score for number of cluster from 2 to 20'
    x_label = 'Number of cluster'
    y_label = 'Silhouette score'
    plot_with_text(x, silhouette_score_avg, title, x_label, y_label, best_n_clusters, max_silhouette_score,
                   save_plot_file)
    return best_n_clusters


def create_sample_data(random_state, data, sample_size):
    np.random.seed(random_state)
    indices = np.random.choice(data.shape[0], sample_size, replace=False)
    data_sample = data[indices]
    return data_sample


def plot_clusters(data, n_clusters, random_state, save_file):
    # Apply PCA to reduce the data to two dimensions
    pca = PCA(n_components=2)
    pca_transformed_data = pca.fit_transform(data)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(data)

    # Plot the clustered data
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_transformed_data[:, 0], pca_transformed_data[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.title('KMeans Clustering Results with PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(scatter, label='Cluster')
    plt.show()
    plt.savefig(save_file)


def k_means(input_file, random_state, sample_size, save_plot_file, save_k_means_file):

    df = pd.read_csv(input_file)
    # Preprocess the entire dataset
    preprocessed_data = preprocess_data(df)
    '''
    # Sample the preprocessed data
    sample_data = create_sample_data(random_state, preprocessed_data, sample_size)
    # Find the best number of clusters using the sample
    n_clusters = find_best_k(sample_data, random_state, save_plot_file)
    # Plot the clustering results for the sample
    '''
    plot_clusters(preprocessed_data, 2, random_state, save_k_means_file)
