import pandas as pd
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from analyse_dataset import compute_correlation
from knn import suppress_values, divide_into_training_and_test_data


def split_dataframe_by_age(input_file, output_file_age_null, output_file_age_not_null):
    df = pd.read_csv(input_file)
    df_age_null = df[df['age'].isnull()]
    df_age_not_null = df[df['age'].notnull()]

    df_age_null.to_csv(output_file_age_null, index=False)
    df_age_not_null.to_csv(output_file_age_not_null, index=False)

    correlation_matrix = df_age_not_null.corr()
    correlation_with_age = correlation_matrix['age']
    sorted_correlations = correlation_with_age.abs().sort_values(ascending=False)
    most_correlated_features = sorted_correlations.index[1:3]
    print(sorted_correlations)
    return [most_correlated_features[0], most_correlated_features[1]]


def create_model_linear_regression(input_file, subset):
    # Split the data into training and testing sets
    data_train, data_test, label_train, label_test = divide_into_training_and_test_data(input_file, 'age',
                                                                                        subset)

    regression = linear_model.LinearRegression()

    regression.fit(data_train, label_train)
    label_predicted = regression.predict(data_test)

    mse = mean_squared_error(label_test, label_predicted)
    print(mse)
    return regression


def split_by_age_and_clean(input_file, output_file_age_null, output_file_age_not_null):
    df = pd.read_csv(input_file)

    threshold = 0.9
    cols_to_drop = df.columns[df.isna().mean() > threshold]
    df.drop(cols_to_drop, axis=1, inplace=True)

    df_age_null = df[df['age'].isnull()]
    df_age_not_null = df[df['age'].notnull()]

    df_age_null.to_csv(output_file_age_null, index=False)
    df_age_not_null.to_csv(output_file_age_not_null, index=False)
    correlation_matrix = df_age_not_null.corr()
    correlation_with_age = correlation_matrix['age']
    sorted_correlations = correlation_with_age.abs().sort_values(ascending=False)
    most_correlated_features = sorted_correlations.index[1:5]
    print(sorted_correlations)
    return [most_correlated_features[0], most_correlated_features[1], most_correlated_features[2],
            most_correlated_features[3]]
