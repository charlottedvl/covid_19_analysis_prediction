from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import pandas as pd

from analyse_dataset import scatter_plots, perform_pca, plot_pca_projection, analyze_dataset, get_distinct_values, \
    compute_correlation
from format_data import format_dataset
from predictions import prediction_recovery, conditional_probability, create_bayesian_network

# File path
file_path = './data/latestdata.csv'
cleaned_file = './data/cleaned_dataset.csv'

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

# Predict P(having_symptoms | visited_Wuhan = yes)
probability_symptoms = conditional_probability(cleaned_file, ['visited_Wuhan'], [1], 'have_symptoms', 1)
print(f'The probability to have symptoms if the person visited Wuhan is : {probability_symptoms}')

# Predict P(true_patient = true | visited_Wuhan = yes, have_symptoms = yes)
probability_true_patient = conditional_probability(cleaned_file, ['visited_Wuhan', 'have_symptoms'], [1, 1],
                                                   'date_confirmation')
print(f'The probability to be a true patient if the person visited Wuhan and has symptoms is : {probability_true_patient}')

# Predict P(die | visited_Wuhan = yes)
probability_die_Wuhan = conditional_probability(cleaned_file, ['visited_Wuhan'], [1], 'outcome', 0)
print(f'The probability to die if the person visited Wuhan is : {probability_die_Wuhan}')

# Create Bayesian network and variable elimination object
query = create_bayesian_network(cleaned_file)

# Predict P(having_symptoms | visited_Wuhan = yes)
probability_symptoms = query.query(variables=['have_symptoms'], evidence={'visited_Wuhan': 1})
print(probability_symptoms)

# Predict P(true_patient = true | visited_Wuhan = yes, have_symptoms = yes)
probability_true_patient = query.query(variables=['is_true_patient'], evidence={'visited_Wuhan': 1, 'have_symptoms': 1})
print(probability_true_patient)

# Predict P(die | visited_Wuhan = yes)
probability_die_Wuhan = query.query(variables=['outcome'], evidence={'visited_Wuhan': 1})
print(probability_die_Wuhan)

# Estimate time to recover from covid
prediction_recovery(cleaned_file)
