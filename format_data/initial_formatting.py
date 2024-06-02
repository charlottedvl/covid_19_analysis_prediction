from datetime import datetime
import pandas as pd


def format_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    remove_rows(df)
    remove_columns(df)
    format_sex(df)
    format_age(df)
    dates_columns = ['date_confirmation', 'date_admission_hospital', 'travel_history_dates', 'date_death_or_discharge']
    for column in dates_columns:
        format_dates(df, column)
    format_chronic_disease_binary(df)
    format_outcome_recovery_time(df)
    format_rename_date_column(df, 'date_onset_symptoms', 'have_symptoms')
    format_rename_date_column(df, 'date_confirmation', 'is_true_patient')
    create_visited_wuhan(df)
    df.to_csv(output_file, index=False)


def remove_rows(dataframe):
    dataframe.dropna(subset=['outcome'], inplace=True)
    print('Rows removed')


def remove_columns(dataframe):
    dataframe.drop(columns=['source', 'chronic_disease', 'sequence_available', 'notes_for_discussion', 'symptoms',
                            'additional_information', 'reported_market_exposure', 'admin1', 'admin2', 'admin3',
                            'admin_id', 'location', 'ID', 'city', 'country', 'province', 'geo_resolution',
                            'data_moderator_initials', 'country_new', 'travel_history_binary', 'latitude',
                            'longitude'], inplace=True)
    print('Columns removed')


def format_age(dataframe):
    for index, value in dataframe['age'].items():
        # If '-' is present, it means that the age is a range
        if '-' in str(value):
            try:
                inferior_age, superior_age = map(int, value.split('-'))
                if not superior_age:
                    dataframe.at[index, 'age'] = inferior_age
                else:
                    dataframe.at[index, 'age'] = (inferior_age + superior_age) / 2
            except ValueError as e:
                inferior_age = int(value[:2])
                dataframe.at[index, 'age'] = inferior_age
    print('Column age formatted')


def format_sex(dataframe):
    for index, value in dataframe['sex'].items():
        if 'female' in str(value):
            dataframe.at[index, 'sex'] = int(0)
        elif 'male' in str(value):
            dataframe.at[index, 'sex'] = int(1)
        else:
            dataframe.at[index, 'sex'] = None
    print('Column sex formatted')


def format_outcome_recovery_time(dataframe):
    dataframe['outcome'] = dataframe['outcome'].astype(str).str.lower()

    death_mask = dataframe['outcome'].str.contains('death|dead|deceased|died', na=False)
    recovered_mask = dataframe['outcome'].str.contains(
        'recovered|recovering|discharged|discharge|released|not hospitalized', na=False)
    other_mask = ~death_mask & ~recovered_mask

    dataframe.loc[death_mask, 'outcome'] = 0
    dataframe.loc[recovered_mask, 'outcome'] = 2
    dataframe.loc[other_mask, 'outcome'] = 1

    valid_recovery_mask = dataframe['date_death_or_discharge'].notnull() & dataframe['date_confirmation'].notnull()
    dataframe.loc[valid_recovery_mask, 'recovery_time'] = dataframe.loc[
                                                              valid_recovery_mask, 'date_death_or_discharge'] - \
                                                          dataframe.loc[valid_recovery_mask, 'date_confirmation']

    dataframe.drop(columns=['date_death_or_discharge'], inplace=True)
    print('Column outcome formatted')


def format_dates(dataframe, column_name):
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str.strip(), "%d.%m.%Y")
        except ValueError:
            return None

    def average_dates(date_str):
        if '-' in date_str:
            if ' - ' in date_str:
                dates = date_str.split(' - ')
            else:
                dates = date_str.split('-')
            date1 = parse_date(dates[0])
            date2 = parse_date(dates[1])
            if date1 and date2:
                avg_date = date1 + (date2 - date1) / 2
                return avg_date.timestamp()
            elif date1:
                return date1.timestamp()
            elif date2:
                return date2.timestamp()
        else:
            date = parse_date(date_str)
            return date.timestamp() if date else None
        return None

    dataframe[column_name] = dataframe[column_name].apply(lambda x: average_dates(str(x)) if pd.notnull(x) else None)

    print(
        f"Column '{column_name}' formatted")


def create_visited_wuhan(dataframe):
    dataframe["visited_Wuhan"] = dataframe.apply(lambda row: 1 if row['lives_in_Wuhan'] == 'yes' or (
            pd.notnull(row['travel_history_location']) and 'Wuhan' in row['travel_history_location']) else 0,
                                                 axis=1)
    dataframe.drop(columns=['lives_in_Wuhan', 'travel_history_location'], inplace=True)
    print('Column visited_Wuhan created')


def format_rename_date_column(dataframe, column, new_name):
    dataframe[column] = dataframe.apply(
        lambda row: 1 if pd.notnull(row[column]) else 0,
        axis=1)
    dataframe.rename(columns={
        column: new_name
    }, inplace=True)
    print(f'Column {new_name} created')


def format_chronic_disease_binary(dataframe):
    dataframe['chronic_disease_binary'] = dataframe.apply(
        lambda row: 1 if str(row['chronic_disease_binary']) == 'True' else 0,
        axis=1)
    print('Column chronic_disease_binary formatted')
