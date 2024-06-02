import pandas as pd

from analyse_dataset import scatter_plots, perform_pca, plot_pca_projection, analyze_dataset, get_distinct_values, \
    compute_correlation, variance_column, percentage_value_column
from clustering_data import k_means
from format_data.initial_formatting import format_dataset
from format_data.process_data import suppress_values, process_data, split_dataframe
from knn import choose_best_k
from bayesian_network import prediction_recovery, create_bayesian_network
from linear_regression import create_model_linear_regression, find_features

# Dataset's files path
file_path = './data/latestdata.csv'
cleaned_file = './data/cleaned_dataset.csv'
dead_discharged_file = 'data/knn_datasets/dead_discharged_dataset.csv'
hospitalized_file = 'data/knn_datasets/hospitalized_dataset.csv'
hospitalized_outcomes_predicted_file = './data/knn_datasets/hospitalized_outcomes_predicted.csv'
age_null_file = 'data/linear_regression_datasets/age_null_dataset.csv'
age_not_null_file = 'data/linear_regression_datasets/age_not_null_dataset.csv'
age_null_cleaned_file = 'data/linear_regression_datasets/age_null_cleaned_dataset.csv'
age_not_null_cleaned_file = 'data/linear_regression_datasets/age_not_null_cleaned_dataset.csv'
predicted_age_file = 'data/linear_regression_datasets/predicted_age.csv'

# Plot file
knn_plot = './data/plots/knn_plot.png'
save_plot_file = 'data/plots/silhouette_score_plot.png'
save_file = 'data/plots/k_means_plot.png'


# Format the dataset
format_dataset(file_path, cleaned_file)

# Extract datatypes and number of empty rows for each column
analyze_dataset(cleaned_file)

# Compute correlation between variables
compute_correlation()

# Scatter plots
scatter_plots('./data/cleaned_dataset.csv')

# PCA analysis
pca_result = perform_pca(cleaned_file)
plot_pca_projection(pca_result, './data/pca_plot.png')


# Create Bayesian network and variable elimination object
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


split_dataframe(cleaned_file, 'outcome', hospitalized_file, dead_discharged_file, values=[0, 2])

# Suppress recovery time has only discharged patients has one
dead_discharged_patients, recovery_times = suppress_values(dead_discharged_file, 'recovery_time')
dead_discharged_patients.to_csv(dead_discharged_file)

hospitalized_patients, recovery_times_bis = suppress_values(hospitalized_file, 'recovery_time')
hospitalized_patients.to_csv(hospitalized_file)

best_k, knn_model = choose_best_k(dead_discharged_file, knn_plot)
data, outcomes = suppress_values(hospitalized_file, 'outcome')
data_processed = process_data('mean', data)
# A bit long to perform
data['outcomes_predicted'] = knn_model.predict(data_processed)
data.to_csv(hospitalized_outcomes_predicted_file)

percentage_value_column(dead_discharged_file, 'outcome')

percentage_value_column(hospitalized_outcomes_predicted_file, 'outcomes_predicted')

# Correct percentage
split_dataframe(cleaned_file, 'age', age_null_file, age_not_null_file)

features = find_features(age_not_null_file, 'age', 2)

model = create_model_linear_regression(age_not_null_file, features)

# Suppress some features that have a lot of missing values and add more observed values to the regression
split_dataframe(cleaned_file, 'age', age_null_cleaned_file, age_not_null_cleaned_file, 0.9)
features_improved = find_features(age_not_null_cleaned_file, 'age', 3)

model_improved = create_model_linear_regression(age_not_null_cleaned_file, features_improved)

# Predict age for missing values
data, labels = suppress_values(age_null_cleaned_file, 'age')
data = data[features_improved]
predicted_age = model_improved.predict(data)
rounded_age_predicted = predicted_age.round(1)

data_with_predicted_age = pd.concat([data, pd.DataFrame(rounded_age_predicted, columns=['predicted_age'])], axis=1)
data_with_predicted_age.to_csv(predicted_age_file, index=False)

# Analyze results ==> very low variance, it isn't much good
variance_column(predicted_age_file, 'predicted_age')


sample_size = 10000
random_state = 10
k_means(cleaned_file, random_state, sample_size, save_plot_file, save_file)

