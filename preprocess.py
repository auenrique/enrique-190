import pandas as pd

data = pd.read_csv('dev.tsv', sep='\t')

#Parse "label" as array of strings
data['label'] = data['label'].apply(lambda x: x.split(','))

#Extract the first label for each example
data['first_label'] = data['label'].apply(lambda x: x[0])

print(data['first_label'])