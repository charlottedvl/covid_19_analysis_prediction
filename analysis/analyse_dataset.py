import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import pandas as pd


def analyze_dataset(csv_file, save_file=None):
    df = pd.read_csv(csv_file)
    #empty_rows_by_column(df)
    #compute_correlation(df)
    #scatter_plots(df)
    features = find_most_correlated(csv_file, 'outcome', 4)
    perform_pca(df, features, save_file)


def empty_rows_by_column(dataframe):
    total_rows = len(dataframe)
    empty_counts = dataframe.isnull().sum()
    column_types = dataframe.dtypes

    print('Column name (column type) : number of empty rows')

    max_col_len = max(len(col) for col in dataframe.columns)
    max_type_len = max(len(str(column_types[col])) for col in dataframe.columns)

    for column in dataframe.columns:
        col_name = column.ljust(max_col_len)
        col_type = str(column_types[column]).ljust(max_type_len)
        empty_count = str(empty_counts[column]).rjust(5)
        print(f'{col_name} ({col_type}): {empty_count} empty')

    print('\nTotal number of rows:')
    print(total_rows)


def compute_correlation(dataframe):
    correlation_matrix = dataframe.corr()

    print("Matrice de corrélation :")
    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            correlation_value = correlation_matrix.loc[col1, col2]
            print(f"{col1} - {col2}: {correlation_value}")


def scatter_plots(dataframe):
    columns = dataframe.columns

    for column1 in columns:
        with PdfPages(f'./data/scatter_plots/scatter_plot_{column1}.pdf') as pdf:
            for column2 in columns:
                if column1 == column2:
                    continue
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(dataframe[column1], dataframe[column2], marker='.')
                ax.set_xlabel(column1)
                ax.set_ylabel(column2)
                plt.title(f'Scatter plot: {column1} vs {column2}')
                pdf.savefig(fig)
                plt.close(fig)
        print(f'Saved scatter plots for {column1} in scatter_plot_{column1}.pdf')


def perform_pca(dataframe, features, save_file):
    feature_data = dataframe[features]

    imputer = SimpleImputer(strategy='most_frequent')
    imputed_data = imputer.fit_transform(feature_data)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    outcome = dataframe['outcome']

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Outcome'] = outcome

    vectors = pca.components_.T

    return plot_pca_projection(pca_df, vectors, features, save_file)


def plot_pca_projection(pca_df, vectors, features, save_file=None):
    plt.figure(figsize=(10, 7))
    scale_factor = 80
    scaled_vectors = vectors * scale_factor

    for i, feature in enumerate(features):
        plt.arrow(0, 0, scaled_vectors[i, 0], scaled_vectors[i, 1],
                  color='red', alpha=0.5, head_width=5)
        plt.text(scaled_vectors[i, 0] * 1.1, scaled_vectors[i, 1] * 1.1, features[i], color='black', ha='center',
                 va='center')

    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Outcome'], cmap='viridis')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection with Feature Vectors')
    plt.grid()

    if save_file:
        plt.savefig(save_file)
        print(f'Plot saved as {save_file}')
    else:
        plt.show()


def find_most_correlated(input_file, column, number_of_features):
    df = pd.read_csv(input_file)

    correlation_matrix = df.corr()
    correlation_with_age = correlation_matrix[column]

    sorted_correlations = correlation_with_age.abs().sort_values(ascending=False)
    most_correlated_features = sorted_correlations.index[1:(number_of_features + 1)]
    return most_correlated_features
