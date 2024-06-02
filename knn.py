from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from format_data.process_data import divide_training_test_data
from plot.plot import plot_with_text


def knn_create_model(data_train, label_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data_train, label_train)
    return knn


def accuracy_model(predicted_values, actual_values):
    accuracy = sum(predicted_values == actual_values) / len(actual_values) * 100
    return accuracy


def choose_best_k(input_file, save_file):
    data_train, data_test, outcomes_train, outcomes_test = divide_training_test_data(input_file, 'outcome')
    x = range(5, 31)
    accuracies = []
    # Iterate through different value of k
    for k in x:
        knn_model = knn_create_model(data_train, outcomes_train, k)
        outcomes_pred = knn_model.predict(data_test)
        accuracies.append(accuracy_model(outcomes_pred, outcomes_test))
        print(k)

    max_accuracy = max(accuracies)
    best_k = accuracies.index(max_accuracy) + 1
    knn_model = knn_create_model(data_train, outcomes_train, best_k)
    outcomes_pred = knn_model.predict(data_test)

    # Plot the results
    title = 'Accuracy percentage for K in range(1, 31)'
    x_label = 'K'
    y_label = 'Accuracy Percentage'
    plot_with_text(x, accuracies, title, x_label, y_label, best_k, max_accuracy, save_file)

    print(f'Max accuracy: {max_accuracy:.2f}' + "%")
    confusion(outcomes_pred, outcomes_test)
    return best_k, knn_model


def confusion(predicted_values, actual_values):
    conf_matrix = confusion_matrix(actual_values, predicted_values)
    print('Confusion Matrix:')
    print(conf_matrix)
