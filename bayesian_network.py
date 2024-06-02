import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork


def conditional_probability(input_file, condition_columns, condition_values, target_column, target_value=None):
    df = pd.read_csv(input_file)
    observed_total = 0
    total = 0

    for index, value in df[target_column].items():
        if all(pd.notnull(df.at[index, column]) and (condition_value is None or df.at[index, column] == condition_value)
               for column, condition_value in zip(condition_columns, condition_values)):
            total += 1
            if pd.notnull(value) and (target_value is None or value == target_value):
                observed_total += 1

    if total == 0:
        return 0
    return (observed_total / total) * 100


def create_bayesian_network(dataset_file):
    df = pd.read_csv(dataset_file)
    model = BayesianNetwork([('visited_Wuhan', 'have_symptoms'),
                             ('visited_Wuhan', 'outcome'),
                             ('have_symptoms', 'outcome'),
                             ('visited_Wuhan', 'is_true_patient'),
                             ('have_symptoms', 'is_true_patient')])

    model.fit(df, estimator=MaximumLikelihoodEstimator)

    query = VariableElimination(model)
    return query


def prediction_recovery(input_file):
    df = pd.read_csv(input_file)
    observed_total = 0
    total = 0

    filtered_df = df[df['visited_Wuhan'] == 1]

    average_recovery_time = filtered_df['recovery_time'].mean()

    mon, seconds = divmod(average_recovery_time, 60)
    hr, minutes = divmod(mon, 60)
    days, hours = divmod(hr, 24)

    return print(f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
