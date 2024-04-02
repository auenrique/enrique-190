import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
import nltk

import tokenizer

def main():
    print('hej')
    data = pd.read_csv('en-annotated.tsv', sep='\t', names=['text', 'label'])
    data.head()

    # Check if there are any missing values.
    data.isnull().sum()

    after_tokenize = tokenizer.tokenize_data(data, True)

    df = pd.DataFrame.from_dict(after_tokenize)

    df.to_csv('after_tokenize.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()