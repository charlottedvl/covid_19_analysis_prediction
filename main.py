# Chemin du fichier CSV
from format_data import remove_empty_age_rows, analyze_csv, compute_mean_age, count_empty_lines_per_column

file_path = './data/latestdata.csv'

output_file = './data/newdata.csv'
new_file = './data/agemean.csv'

remove_empty_age_rows(file_path, output_file)
#analyze_csv(output_file)
compute_mean_age(output_file, new_file)
count_empty_lines_per_column(new_file)