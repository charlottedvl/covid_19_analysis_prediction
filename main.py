import pandas as pd

from analysis.analyse_column import percentage_value_column, analyse_column
from analysis.analyse_dataset import analyze_dataset, compute_correlation, find_most_correlated
from models.grid_search import grid_search
from models.k_means import k_means
from format_data.initial_formatting import format_dataset
from format_data.process_data import suppress_values, process_data, split_dataframe
from models.knn import choose_best_k
from models.bayesian_network import prediction_recovery, create_bayesian_network
from models.linear_regression import create_model_linear_regression

# Dataset's files path
original_dataset_file = './data/latestdata.csv'
cleaned_file = './data/cleaned_dataset.csv'
dead_discharged_file = 'data/knn/dead_discharged_dataset.csv'
hospitalized_file = 'data/knn/hospitalized_dataset.csv'
hospitalized_outcomes_predicted_file = 'data/knn/hospitalized_outcomes_predicted.csv'
age_null_file = 'data/linear_regression_datasets/age_null_dataset.csv'
age_not_null_file = 'data/linear_regression_datasets/age_not_null_dataset.csv'
age_null_cleaned_file = 'data/linear_regression_datasets/age_null_cleaned_dataset.csv'
age_not_null_cleaned_file = 'data/linear_regression_datasets/age_not_null_cleaned_dataset.csv'
predicted_age_file = 'data/linear_regression_datasets/predicted_age.csv'

# Plot file
pca_plot_file = 'data/analysis/pca_plot.png'
knn_plot_file = 'data/knn/knn_plot.png'
silhouette_score_file = 'data/k_means/silhouette_score_plot.png'
k_means_plot_file = 'data/k_means/k_means_plot.png'

###################################
# Format the dataset

format_dataset(original_dataset_file, cleaned_file)


###################################
# Analyze the dataset

analyze_dataset(cleaned_file, pca_plot_file)


###################################
# Use Bayesian network to compute some probabilities

# Create the bayesian network
bayes_net = create_bayesian_network(cleaned_file)

# Predict P(having_symptoms | visited_Wuhan = yes)
probability_symptoms = bayes_net.query(variables=['have_symptoms'], evidence={'visited_Wuhan': 1})
print(probability_symptoms)

# Predict P(true_patient = true | visited_Wuhan = yes, have_symptoms = yes)
probability_true_patient = bayes_net.query(variables=['is_true_patient'], evidence={'visited_Wuhan': 1, 'have_symptoms': 1})
print(probability_true_patient)

# Predict P(die | visited_Wuhan = yes)
probability_die_Wuhan = bayes_net.query(variables=['outcome'], evidence={'visited_Wuhan': 1})
print(probability_die_Wuhan)

# Estimate time to recover from covid
prediction_recovery(cleaned_file)


###################################
# K-Nearest Neighbors to predict outcomes

# Split the dataset in two file : one for patient whose outcome is known, one for patients whose outcome is not known
split_dataframe(cleaned_file, 'outcome', hospitalized_file, dead_discharged_file, values=[0, 2])

# Suppress recovery time as only discharged patients has one
dead_discharged_patients, recovery_times = suppress_values(dead_discharged_file, 'recovery_time')
dead_discharged_patients.to_csv(dead_discharged_file)

hospitalized_patients, recovery_times_bis = suppress_values(hospitalized_file, 'recovery_time')
hospitalized_patients.to_csv(hospitalized_file)

# Choose best k for the model
best_k, knn_model = choose_best_k(dead_discharged_file, knn_plot_file)

# Suppress outcomes (which are all equal to 1)
data, outcomes = suppress_values(hospitalized_file, 'outcome')
data_processed = process_data('mean', data)

# Predict outcomes
data['outcomes_predicted'] = knn_model.predict(data_processed)
data.to_csv(hospitalized_outcomes_predicted_file)

# Compute the percentage of each outcome in each file
percentage_value_column(dead_discharged_file, 'outcome')
percentage_value_column(hospitalized_outcomes_predicted_file, 'outcomes_predicted')


###################################
# Regression to predict age of the patient

# Split the dataset in two file : one for patient whose age is known, one for patients whose age is not known
split_dataframe(cleaned_file, 'age', age_null_file, age_not_null_file)

# Print the correlation matrix
print(compute_correlation(pd.read_csv(age_not_null_file), 'age'))

# Find the two most correlated features
features = find_most_correlated(age_not_null_file, 'age', 2)

# Create the linear regression according to these features
model = create_model_linear_regression(age_not_null_file, features)

# Suppress some features that have a lot of missing values and add more observed values to the regression
split_dataframe(cleaned_file, 'age', age_null_cleaned_file, age_not_null_cleaned_file, 0.9)

# Print the correlation matrix
print(compute_correlation(pd.read_csv(age_not_null_cleaned_file), 'age'))

# Find the three most correlated features
features_improved = find_most_correlated(age_not_null_cleaned_file, 'age', 3)

# Create the linear regression according to these features
model_improved = create_model_linear_regression(age_not_null_cleaned_file, features_improved)

# Predict age for missing values
data, labels = suppress_values(age_null_cleaned_file, 'age')
data = data[features_improved]
predicted_age = model_improved.predict(data)
rounded_age_predicted = predicted_age.round(1)

# Add predicted_age column to the file
data_with_predicted_age = pd.concat([data, pd.DataFrame(rounded_age_predicted, columns=['predicted_age'])], axis=1)
data_with_predicted_age.to_csv(predicted_age_file, index=False)

# Analyze the mean and the variance of the columns
analyse_column(predicted_age_file, 'predicted_age')
analyse_column(age_not_null_cleaned_file, 'age')


###################################
# K-Means to cluster the dataset

sample_size = 10000
random_state = 10
k_means(cleaned_file, random_state, sample_size, silhouette_score_file, k_means_plot_file)


###################################
# Grid Search to find the best parameters to our models
grid_search(cleaned_file, 'kmeans')
grid_search(cleaned_file, 'knn')
