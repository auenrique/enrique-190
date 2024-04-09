import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import tokenizer
import emotions
import ngrams_svc

def main():
    data = pd.read_csv('en-annotated-ungrouped.tsv', sep='\t')
    data.head()

    # Check if there are any missing values.
    data.isnull().sum()

    after_tokenize = tokenizer.tokenize_data(data, True)

    dd = pd.DataFrame.from_dict(after_tokenize)
    #print(dd.shape)
    #drop columns where preprocessed is empty
    #dd = dd[dd['preprocessed'].map(lambda d: len(d)) > 0]
    #print(dd.shape)

    

    pp = dd['text']
    df = dd['label']

    train_dev_X, test_X, train_dev_y, test_y = train_test_split(pp, df, test_size=0.1, stratify=df, random_state=42)
    train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, stratify=train_dev_y, random_state=42)

    vec = CountVectorizer(ngram_range=(1,3))
    train_X_dtm = vec.fit_transform(train_X)
    test_X_dtm = vec.transform(test_X) 

    X = train_X_dtm
    y = train_y
    #print(train_X_dtm)

    clf = LinearSVC(C=1, random_state=42)
    #clf = KNeighborsClassifier(n_neighbors=3)

    clf.fit(X, y)

    XX = test_X_dtm
    yy = test_y

    y_pred = clf.predict(XX)

    tp = 0
    fp = 0
    fn = 0

    target_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

    print(classification_report(yy, y_pred, target_names=target_names))

if __name__ == "__main__":
    main()
