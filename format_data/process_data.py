import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def process_data(strategy, data_train, data_test=None):
    imputer = SimpleImputer(strategy=strategy)
    data_train = imputer.fit_transform(data_train)
    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    if data_test is None:
        return data_train
    data_test = imputer.transform(data_test)
    data_test = scaler.transform(data_test)
    return data_train, data_test


def divide_training_test_data(input_file, column, strategy='most_frequent', subset=None):
    data, labels = suppress_values(input_file, column)
    if subset is not None:
        data = data[subset]
    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.3, random_state=4)

    data_train, data_test = process_data(strategy, data_train, data_test)
    return data_train, data_test, label_train, label_test


def split_dataframe(input_file, column, output_file_to_predict, output_file_to_evaluate, threshold=None, values=None):
    df = pd.read_csv(input_file)

    if threshold is not None:
        cols_to_drop = df.columns[df.isna().mean() > threshold]
        df.drop(cols_to_drop, axis=1, inplace=True)

    if values is not None:
        df_to_predict = df[~df[column].isin(values)]
        df_to_evaluate_model = df[df[column].isin(values)]
    else:
        df_to_predict = df[df[column].isnull()]
        df_to_evaluate_model = df[df[column].notnull()]

    df_to_predict.to_csv(output_file_to_predict, index=False)
    df_to_evaluate_model.to_csv(output_file_to_evaluate, index=False)


def suppress_values(input_file, column):
    df = pd.read_csv(input_file)

    data = df.drop(column, axis=1)
    labels = df[column]
    return data, labels
