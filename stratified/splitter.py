import pandas as pd
from sklearn.model_selection import train_test_split

# Read the en-annotated.tsv file
data = pd.read_csv('en-annotated.tsv', sep='\t', names=['text', 'label'])

# Split the label column into multiple values
data['labelarr'] = data['label'].str.split(',')

# Extract the first label for each example
data['first_label'] = data['labelarr'].apply(lambda x: x[0])

# Perform stratified train-dev-test split
train_dev, test = train_test_split(data, test_size=0.1, stratify=data['first_label'])
train, dev = train_test_split(train_dev, test_size=0.222, stratify=train_dev['first_label'])

# Print the number of examples in each split
print('train:', len(train))
print('dev:', len(dev))
print('test:', len(test))

# Remove the first_label and labelarr column
# train = train.drop(columns=['first_label', 'labelarr'])
# dev = dev.drop(columns=['first_label', 'labelarr'])
# test = test.drop(columns=['first_label', 'labelarr'])

# Save the splits to separate tsv files
train.to_csv('train.tsv', sep='\t', index=False)
dev.to_csv('dev.tsv', sep='\t', index=False)
test.to_csv('test.tsv', sep='\t', index=False)

