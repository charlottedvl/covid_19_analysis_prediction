import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


import pandas as pd


def get_distinct_values(input_file, column_name):
    df = pd.read_csv(input_file)

    # Get distinct values and their counts in the specified column
    value_counts = df[column_name].dropna().value_counts()

    # Print distinct values and their counts
    print(f"Distinct values in column '{column_name}':")
    for value, count in value_counts.items():
        print(f"{value}: {count} occurrences")


def analyze_dataset(csv_file):
    df = pd.read_csv(csv_file, delimiter=',')

    total_rows = len(df)
    empty_counts = df.isnull().sum()
    column_types = df.dtypes

    print('Column name (column type) : number of empty rows')

    max_col_len = max(len(col) for col in df.columns)
    max_type_len = max(len(str(column_types[col])) for col in df.columns)

    for column in df.columns:
        col_name = column.ljust(max_col_len)
        col_type = str(column_types[column]).ljust(max_type_len)
        empty_count = str(empty_counts[column]).rjust(5)
        print(f'{col_name} ({col_type}): {empty_count} empty')

    print('\nTotal number of rows:')
    print(total_rows)


def compute_correlation(input_file):
    df = pd.read_csv(input_file)

    correlation_matrix = df.corr()

    print("Matrice de corrélation :")
    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            correlation_value = correlation_matrix.loc[col1, col2]
            print(f"{col1} - {col2}: {correlation_value}")


def scatter_plots(file):
    df = pd.read_csv(file)
    columns = df.columns

    for column1 in columns:
        with PdfPages(f'./data/scatter_plots/scatter_plot_{column1}.pdf') as pdf:
            for column2 in columns:
                if column1 == column2:
                    continue
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(df[column1], df[column2], marker='.')
                ax.set_xlabel(column1)
                ax.set_ylabel(column2)
                plt.title(f'Scatter plot: {column1} vs {column2}')
                pdf.savefig(fig)
                plt.close(fig)
        print(f'Saved scatter plots for {column1} in scatter_plot_{column1}.pdf')


def perform_pca(file):
    df = pd.read_csv(file)

    # Impute missing values with the mode (most frequent value) of each column
    imputer = SimpleImputer(strategy='most_frequent')
    imputed_data = imputer.fit_transform(df)

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    return pca_result


def plot_pca_projection(pca_result, save_file=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], marker='.')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection')

    if save_file:
        plt.savefig(save_file)
        print(f'Plot saved as {save_file}')
    else:
        plt.show()


