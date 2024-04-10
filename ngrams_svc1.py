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
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack

import tokenizer
import emotions

def count_labels(data):
    #print(data)
    labels = {}
    for i in range(1,9):
        labels[i] = 0
    for i in data:
        for j in range(1,9):
            if str(j) in i:
                labels[j] += 1
    return labels

def main():
    data = pd.read_csv('en-annotated.tsv', sep='\t', names=['text', 'label'])
    data.head()

    # Check if there are any missing values.
    print(data.isnull().sum())
    
    after_tokenize = tokenizer.tokenize_data(data, True)

    dd = pd.DataFrame.from_dict(after_tokenize)
    dd.to_csv ('help.tsv', sep='\t', index=False)
    #print(dd.shape)
    #drop columns where preprocessed is empty
    #dd = dd[dd['preprocessed'].map(lambda d: len(d)) > 0]
    #print(dd.shape)

    vec = CountVectorizer(ngram_range=(1,3))
    tf = HashingVectorizer(ngram_range=(1,3))
    xd = vec.fit_transform(dd['text'])
    #print datatype of xd
    #print(type(xd))

    #print(count_labels(after_tokenize['label']))
    df = pd.DataFrame.from_dict(dd['label'].values.ravel())

    lex = emotions.build_lexicon()
    #emo_features = emotions.get_emotion_features(after_tokenize, lex)

    #ef = pd.DataFrame.from_dict(emo_features)
    #xd = pd.DataFrame(dataframes['unigram_bigram_trigram'].toarray())
    
    #merge emo_features and n-gram features
    # xd = pd.concat([ef, xd], axis=1)
    #xd = np.concatenate((ef, xd.toarray()), axis=1)
    #xd = np.array(dataframes['unigram_bigram_trigram'].toarray())
    xd = dd['text']

    train_dev_X, test_X, train_dev_y, test_y = train_test_split(xd, df[0], test_size=0.1, stratify=df[0].str.split(',').apply(lambda x: x[0]), random_state=42)
    train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, stratify=train_dev_y.str.split(',').apply(lambda x: x[0]), random_state=42)

    # train_dev_X, test_X, train_dev_y, test_y = train_test_split(xd, df[0], test_size=0.1, random_state=42)
    # train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, random_state=42)
    # ddf = pd.DataFrame(df[0].str.split(',').apply(lambda x: x[0]))
    
    # train_dev_X, test_X, train_dev_y, test_y = train_test_split(xd, ddf[0], test_size=0.1, stratify=ddf[0], random_state=42)
    # train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, stratify=train_dev_y, random_state=42)

    #clf = LinearSVC(class_weight='balanced', C=0.05, random_state=42)
    train_X = train_X.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)

    tok_Xtrain = train_X.apply(lambda x: tokenizer.tokenize_nostem(x))
    tok_Xtest = test_X.apply(lambda x: tokenizer.tokenize_nostem(x))

    emo_Xtrain = emotions.get_emotion_features(tok_Xtrain, lex)
    emo_Xtest = emotions.get_emotion_features(tok_Xtest, lex)

    clf = OneVsRestClassifier(LinearSVC(class_weight='balanced', C=0.05, random_state=42))

    vec = CountVectorizer(analyzer='word',ngram_range=(1,3))
    mlb = MultiLabelBinarizer()

    X = vec.fit_transform(train_X)
    y = mlb.fit_transform(train_y.str.replace(' ', '').str.split(','))
    

    #append emotion features to n-gram features
    X = hstack([X, pd.DataFrame(emo_Xtrain)])

    #print(X.shape, y.shape)

    clf.fit(X, y)

    XX = vec.transform(test_X)
    yy = mlb.transform(test_y.str.replace(' ', '').str.split(','))

    XX = hstack([XX, pd.DataFrame(emo_Xtest)])
    test = clf.predict(XX)

    #print(set(yy.ravel()) - set(test.ravel()))

    target_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    #print(multilabel_confusion_matrix(yy, test),target_names)
    print(classification_report(yy, test, target_names=target_names))
    #print(confusion_matrix(yy, test))

    #todo
    #try other countvectorizers
    #look at other papers that use xed
   

if __name__ == "__main__":
    main()