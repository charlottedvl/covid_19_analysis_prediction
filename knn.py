import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def split_dataframe_by_outcome(input_file, output_file_1, output_file_2):
    df = pd.read_csv(input_file)

    df_dead_discharge = df[df['outcome'].isin([2, 0])]
    df_hospitalized = df[df['outcome'] == 1]

    df_dead_discharge.to_csv(output_file_1, index=False)
    df_hospitalized.to_csv(output_file_2, index=False)


def suppress_values(input_file, column):
    df = pd.read_csv(input_file)

    data = df.drop(column, axis=1)
    labels = df[column]
    return data, labels


def divide_into_training_and_test_data(input_file, column, subset=None):
    data, labels = suppress_values(input_file, column)
    if subset is not None:
        data = data[subset]
    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.3, random_state=4)

    imputer = SimpleImputer(strategy='most_frequent')
    data_train = imputer.fit_transform(data_train)
    data_test = imputer.transform(data_test)

    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    return data_train, data_test, label_train, label_test


def knn_create_model(data_train, label_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data_train, label_train)
    return knn


def accuracy_model(predicted_values, actual_values):
    accuracy = sum(predicted_values == actual_values) / len(actual_values) * 100
    return accuracy


def choose_best_k(input_file, save_file):
    data_train, data_test, outcomes_train, outcomes_test = divide_into_training_and_test_data(input_file, 'outcomes')

    accuracies = []
    # Iterate through different value of k
    for k in range(1, 51):
        knn_model = knn_create_model(data_train, outcomes_train, k)

        # KNN prediction
        outcomes_pred = knn_model.predict(data_test)

        # Error percentage
        accuracies.append(accuracy_model(outcomes_pred, outcomes_test))
        print(k)

    max_accuracy = max(accuracies)
    best_k = accuracies.index(max_accuracy) + 1
    knn_model = knn_create_model(data_train, outcomes_train, best_k)
    outcomes_pred = knn_model.predict(data_test)

    # Plot the results
    plt.plot(range(1, 51), accuracies, color='blue', linestyle='dashed', markerfacecolor='red', markersize=10)
    plt.title('Error Percentage for K in range(1, 51)')
    plt.xlabel('K')
    plt.ylabel('Accuracy Percentage')
    plt.scatter(best_k, max_accuracy, color='green', s=100)
    plt.text(best_k, max_accuracy, f'k={best_k}, Acc={max_accuracy:.2f}%', fontsize=12, verticalalignment='bottom')
    plt.savefig(save_file)

    print(f'Accuracy: {max_accuracy:.2f}' + "%")
    confusion_matrix(outcomes_pred, outcomes_test)
    return best_k, knn_model


def confusion(predicted_values, actual_values):
    conf_matrix = confusion_matrix(actual_values, predicted_values)
    print('Confusion Matrix:')
    print(conf_matrix)
