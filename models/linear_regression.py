import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from format_data.process_data import divide_training_test_data


def create_model_linear_regression(input_file, subset):
    # Split the data into training and testing sets
    data_train, data_test, label_train, label_test = divide_training_test_data(input_file, 'age', subset=subset)
    data_train = pd.DataFrame(data_train, columns=subset)
    data_test = pd.DataFrame(data_test, columns=subset)
    label_train = pd.DataFrame(label_train, columns=['age'])
    label_test = pd.DataFrame(label_test, columns=['age'])

    regression = linear_model.LinearRegression()

    regression.fit(data_train, label_train)
    label_predicted = regression.predict(data_test)

    mse = mean_squared_error(label_test, label_predicted)
    print(mse)
    return regression
