import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from format_data.process_data import divide_training_test_data


def grid_search(input_file, model_type):
    if model_type == 'knn':
        data_train, data_test, label_train, label_test = divide_training_test_data(input_file, 'outcome')
        param_grid = {
            'n_neighbors': [1, 2],
            'metric': ['euclidean']
        }
        model = KNeighborsClassifier()
        scoring = 'accuracy'
    elif model_type == 'kmeans':
        data_train = pd.read_csv(input_file)
        param_grid = {'n_clusters': [2, 3]}
        model = KMeans(random_state=42)
        scoring = 'silhouette_score'
    else:
        raise ValueError("Unsupported model type. Choose from 'knn' or 'kmeans'.")
    print('gred')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring)

    if model_type in ['knn']:
        print("hnn")
        grid_search.fit(data_train, label_train)
    else:
        grid_search.fit(data_train)

    return grid_search.best_params_, grid_search.best_score_
