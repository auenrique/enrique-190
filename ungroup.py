import pandas as pd
from sklearn.model_selection import train_test_split

# Read the en-annotated.tsv file
data = pd.read_csv('en-annotated.tsv', sep='\t', names=['text', 'label'])

#iterate through each row
lst = []
for index, row in data.iterrows():
    labelarr = row['label'].split(',')
    for i in range(0, len(labelarr)):
        new_row = row.copy()
        new_row['label'] = labelarr[i]
        lst.append(new_row)

new_data = pd.DataFrame(lst)    

new_data.to_csv('en-annotated-ungrouped.tsv', sep='\t', index=False)