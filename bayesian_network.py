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
    for index, value in df['outcome'].items():
        if pd.notnull(df.at[index, 'visited_Wuhan']) and df.at[index, 'visited_Wuhan'] == 1 and \
                pd.notnull(df.at[index, 'date_death_or_discharge']) and pd.notnull(df.at[index, 'date_confirmation']):
            if pd.notnull(value) and value == 2:
                total += 1
                date_diff = df.at[index, 'date_death_or_discharge'] - df.at[index, 'date_confirmation']
                observed_total += date_diff

    if total == 0:
        return print("No data to calculate the average recovery time.")

    average_recovery_time = observed_total / total

    mon, seconds = divmod(average_recovery_time, 60)
    hr, minutes = divmod(mon, 60)
    days, hours = divmod(hr, 24)

    return print(f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
