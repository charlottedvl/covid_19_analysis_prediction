import pandas as pd


def get_distinct_values(input_file, column_name):
    df = pd.read_csv(input_file)

    # Get distinct values and their counts in the specified column
    value_counts = df[column_name].dropna().value_counts()

    # Print distinct values and their counts
    print(f"Distinct values in column '{column_name}':")
    for value, count in value_counts.items():
        print(f"{value}: {count} occurrences")


def analyse_column(input_file, column):
    df = pd.read_csv(input_file)
    variance = df[column].var()
    mean = df[column].mean()
    print(f'The mean of the column {column} is {mean}')
    return print(f'The variance of the column {column} is {variance}')


def percentage_value_column(input_file, column):
    df = pd.read_csv(input_file)
    value_counts = df[column].value_counts(normalize=True) * 100
    return print(value_counts)
