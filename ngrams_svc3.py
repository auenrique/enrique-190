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
import emotions
import ngrams_svc

def main():
    data = pd.read_csv('mtsamples.csv')
    data.head()

    # Check if there are any missing values.
    #print(data.isnull().sum())

    data = data.dropna(subset=['transcription', 'medical_specialty'])
    data.columns = data.columns.str.strip()
    df_obj = data.select_dtypes(['object'])
    data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    data = data[data['medical_specialty'].isin(['Neurosurgery','ENT - Otolaryngology','Discharge Summary'])]
    print(data.shape)

    after_tokenize = dict()

    stopword = nltk.corpus.stopwords.words('english')
    lemmatizer = nltk.WordNetLemmatizer()
    after_tokenize["text"] = data["transcription"]
    after_tokenize["tokenized"] = after_tokenize["text"].apply(lambda x: tokenizer.tokenize_stream(x, stopword, lemmatizer))
    after_tokenize["preprocessed"] = after_tokenize["tokenized"].apply(lambda x: tokenizer.preprocess_stream(x))

    le = LabelEncoder()
    le.fit(data['medical_specialty'])
    after_tokenize['label'] = le.transform(data['medical_specialty'])
    print(data['medical_specialty'].unique())

    #after_tokenize = tokenizer.tokenize_data(data, True)


    #after_tokenize['first_label'] = after_tokenize['label'].str.split(',').apply(lambda x: x[0])

    pp = pd.DataFrame.from_dict(after_tokenize['preprocessed'].values.ravel())
    df = pd.DataFrame.from_dict(after_tokenize['label'])

    # hasone = df[0].str.contains('1').astype(int)
    print (pp[0])
    print(df[0])

    train_dev_X, test_X, train_dev_y, test_y = train_test_split(pp[0], df[0], test_size=0.1, stratify=df[0], random_state=42)
    train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, stratify=train_dev_y, random_state=42)

    vec = CountVectorizer(ngram_range=(1,1))
    train_X_dtm = vec.fit_transform(train_X)
    test_X_dtm = vec.transform(test_X) 

    X = train_X_dtm
    y = train_y
    #print(train_X_dtm)

    clf = LinearSVC(class_weight='balanced', C=1.0, random_state=42)

    clf.fit(X, y)

    XX = test_X_dtm
    yy = test_y

    y_pred = clf.predict(XX)

    tp = 0
    fp = 0
    fn = 0

    print(clf.score(XX, yy))
    #get f1 score
    print(f1_score(yy, y_pred, average='weighted'))

if __name__ == "__main__":
    main()
