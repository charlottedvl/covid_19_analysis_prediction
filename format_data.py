import pandas as pd
from numpy import mean


def analyze_csv(file_path):
    # Lire le fichier CSV
    df = pd.read_csv(file_path)

    # Obtenir les noms des colonnes
    column_names = df.columns.tolist()

    # Déterminer les types des colonnes
    column_types = df.dtypes.to_dict()

    # Afficher les résultats
    for column in column_names:
        print(f"Colonne: {column}, Type: {column_types[column]}")


def remove_empty_age_rows(input_file, output_file):
    # Lire le fichier CSV
    df = pd.read_csv(input_file)

    # Supprimer les lignes où la colonne 'age' est vide (NaN)
    df = df.dropna(subset=['age'])
    df = df.dropna(subset=['outcome'])
    df = df.dropna(subset=['date_confirmation'])

    df = df.drop(columns=['source', 'chronic_disease', 'sequence_available', 'notes_for_discussion'])

    # Sauvegarder le DataFrame nettoyé dans un nouveau fichier CSV
    df.to_csv(output_file, index=False)


def compute_mean_age(input_file, output_file):
    df = pd.read_csv(input_file)

    for index, value in df['age'].items():

        if '-' in str(value):
            try:
                print(value + " " + str(index))
                inferior_age, superior_age = map(int, value.split('-'))
                if not superior_age:
                    df.at[index, 'age'] = inferior_age
                else:
                    df.at[index, 'age'] = (inferior_age + superior_age) / 2
                print(df.at[index, 'age'])
            except ValueError as e:
                inferior_age = int(value[:2])
                df.at[index, 'age'] = inferior_age

                print(f"Erreur de conversion des valeurs à la ligne {index}, value: {value}")
    df.to_csv(output_file, index=False)

def count_empty_lines_per_column(csv_file):
    df = pd.read_csv(csv_file, delimiter=',')
    total_rows = len(df)
    empty_counts = df.isnull().sum()
    print('nombre de lignes vides par colonnes: ')
    print(empty_counts)
    print('total de lignes : ')
    print(total_rows)
