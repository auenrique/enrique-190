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

def train_model(data, use_intensity):
    clf = OneVsRestClassifier(LinearSVC(dual=True, class_weight='balanced', C=0.05, random_state=42))
    classification_reports = []

    data.head()
    data = data.reset_index(drop=True)

    x = data['text']
    y = data['label']
    #print(X.shape, y.shape)

    for train_index, test_index in StratifiedKFold(n_splits=10, random_state=42, shuffle=True).split(x, y.str.replace(' ', '').str.split(',').apply(lambda x: x[0])):
        train_X, test_X = x[train_index], x[test_index]
        train_y, test_y = y[train_index], y[test_index]

        train_X = train_X.reset_index(drop=True)
        test_X = test_X.reset_index(drop=True)

        preprocess_Xtrain = tokenizer.tokenize_data(train_X, True)["preprocessed"]
        preprocess_Xtest = tokenizer.tokenize_data(test_X, True)["preprocessed"]

        vec = CountVectorizer(analyzer='word',ngram_range=(1,3))
        mlb = MultiLabelBinarizer() 

        X = vec.fit_transform(preprocess_Xtrain)
        y = mlb.fit_transform(train_y.str.replace(' ', '').str.split(','))

        XX = vec.transform(preprocess_Xtest)
        yy = mlb.transform(test_y.str.replace(' ', '').str.split(','))

        if(use_intensity):
            lex = emotions.build_lexicon()
            tok_Xtrain = train_X.apply(lambda x: tokenizer.tokenize_nostem(x))
            tok_Xtest = test_X.apply(lambda x: tokenizer.tokenize_nostem(x))

            emo_Xtrain = emotions.get_emotion_features(tok_Xtrain, lex)
            emo_Xtest = emotions.get_emotion_features(tok_Xtest, lex)
            X = hstack([X, pd.DataFrame(emo_Xtrain)])
            XX = hstack([XX, pd.DataFrame(emo_Xtest)])


        tok_Xtrain = train_X.apply(lambda x: tokenizer.tokenize_nostem(x))
        tok_Xtest = test_X.apply(lambda x: tokenizer.tokenize_nostem(x))

        emo_Xtrain = emotions.get_emotion_features(tok_Xtrain, lex)
        emo_Xtest = emotions.get_emotion_features(tok_Xtest, lex)

        clf.fit(X, y)

        test = clf.predict(XX)

        target_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        #print(multilabel_confusion_matrix(yy, test),target_names)
        #print(classification_report(yy, test, target_names=target_names))
        classification_reports.append(classification_report(yy, test, target_names=target_names, output_dict=True, zero_division=1.0))

    print(sum(report['micro avg']['f1-score'] for report in classification_reports)/len(classification_reports))
    print(sum(report['macro avg']['f1-score'] for report in classification_reports)/len(classification_reports))
    print(sum(report['weighted avg']['f1-score'] for report in classification_reports)/len(classification_reports))

def main():
    data = pd.read_csv('en-annotated.tsv', sep='\t', names=['text', 'label'])
    data.head()

    # Check if there are any missing values.
    print(data.isnull().sum())
    
    df = pd.DataFrame.from_dict(data['label'].values.ravel())[0]

    xd = data['text']
    lex = emotions.build_lexicon()

    train_model(data, True)

    # clf = OneVsRestClassifier(LinearSVC(dual=True, class_weight='balanced', C=0.05, random_state=42))

    # classification_reports = []

    # for train_index, test_index in StratifiedKFold(n_splits=10, random_state=42, shuffle=True).split(xd, df.str.replace(' ', '').str.split(',').apply(lambda x: x[0])):
    #     train_X, test_X = xd[train_index], xd[test_index]
    #     train_y, test_y = df[train_index], df[test_index]

    #     train_X = train_X.reset_index(drop=True)
    #     test_X = test_X.reset_index(drop=True)

    #     tok_Xtrain = train_X.apply(lambda x: tokenizer.tokenize_nostem(x))
    #     tok_Xtest = test_X.apply(lambda x: tokenizer.tokenize_nostem(x))

    #     emo_Xtrain = emotions.get_emotion_features(tok_Xtrain, lex)
    #     emo_Xtest = emotions.get_emotion_features(tok_Xtest, lex)

    #     preprocess_Xtrain = tokenizer.tokenize_data(train_X, True)["preprocessed"]
    #     preprocess_Xtest = tokenizer.tokenize_data(test_X, True)["preprocessed"]

    #     vec = CountVectorizer(analyzer='word',ngram_range=(1,3))
    #     mlb = MultiLabelBinarizer() 

    #     #X = vec.fit_transform(train_X)
    #     X = vec.fit_transform(preprocess_Xtrain)
    #     y = mlb.fit_transform(train_y.str.replace(' ', '').str.split(','))

    #     #append emotion features to n-gram features
    #     X = hstack([X, pd.DataFrame(emo_Xtrain)])

    #     #print(X.shape, y.shape)

    #     clf.fit(X, y)

    #     #XX = vec.transform(test_X)
    #     XX = vec.transform(preprocess_Xtest)
    #     yy = mlb.transform(test_y.str.replace(' ', '').str.split(','))

    #     XX = hstack([XX, pd.DataFrame(emo_Xtest)])
    #     test = clf.predict(XX)

    #     target_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    #     #print(multilabel_confusion_matrix(yy, test),target_names)
    #     #print(classification_report(yy, test, target_names=target_names))
    #     classification_reports.append(classification_report(yy, test, target_names=target_names, output_dict=True, zero_division=1.0))


    # print(sum(report['micro avg']['f1-score'] for report in classification_reports)/len(classification_reports))
    # print(sum(report['macro avg']['f1-score'] for report in classification_reports)/len(classification_reports))
    # print(sum(report['weighted avg']['f1-score'] for report in classification_reports)/len(classification_reports))
    # #todo
    # #try other countvectorizers
    # #look at other papers that use xed
   

if __name__ == "__main__":
    main()