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
    
    after_tokenize = tokenizer.tokenize_fromdf(data, True)

    dd = pd.DataFrame.from_dict(after_tokenize)
    dd.to_csv ('help.tsv', sep='\t', index=False)

    df = pd.DataFrame.from_dict(data['label'].values.ravel())[0]

    lex = emotions.build_lexicon()
    xd = data['text']

    train_dev_X, test_X, train_dev_y, test_y = train_test_split(xd, df, test_size=0.1, stratify=df.str.split(',').apply(lambda x: x[0]), random_state=42)
    train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, stratify=train_dev_y.str.split(',').apply(lambda x: x[0]), random_state=42)

    # train_dev_X, test_X, train_dev_y, test_y = train_test_split(xd, df[0], test_size=0.1, random_state=42)
    # train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, random_state=42)
    # ddf = pd.DataFrame(df[0].str.split(',').apply(lambda x: x[0]))
    
    # train_dev_X, test_X, train_dev_y, test_y = train_test_split(xd, ddf[0], test_size=0.1, stratify=ddf[0], random_state=42)
    # train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, stratify=train_dev_y, random_state=42)

    #clf = LinearSVC(class_weight='balanced', C=0.05, random_state=42)
    # test_X = xd
    # test_y = df

    train_X = train_X.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)

    tok_Xtrain = train_X.apply(lambda x: tokenizer.tokenize_nostem(x))
    tok_Xtest = test_X.apply(lambda x: tokenizer.tokenize_nostem(x))

    emo_Xtrain = emotions.get_emotion_features(tok_Xtrain, lex)
    emo_Xtest = emotions.get_emotion_features(tok_Xtest, lex)

    preprocess_Xtrain = tokenizer.tokenize_data(train_X, True)["preprocessed"]
    preprocess_Xtest = tokenizer.tokenize_data(test_X, True)["preprocessed"]

    clf = OneVsRestClassifier(LinearSVC(dual='auto', class_weight='balanced', C=0.01, random_state=42))

    vec = CountVectorizer(analyzer='word',ngram_range=(1,3))
    mlb = MultiLabelBinarizer()

    #X = vec.fit_transform(train_X)
    X = vec.fit_transform(preprocess_Xtrain)
    y = mlb.fit_transform(train_y.str.replace(' ', '').str.split(','))

    #append emotion features to n-gram features
    X = hstack([X, pd.DataFrame(emo_Xtrain)])

    #print(X.shape, y.shape)

    clf.fit(X, y)

    #XX = vec.transform(test_X)
    XX = vec.transform(preprocess_Xtest)
    yy = mlb.transform(test_y.str.replace(' ', '').str.split(','))

    XX = hstack([XX, pd.DataFrame(emo_Xtest)])
    test = clf.predict(XX)

    emo = ['raw_anger', 'raw_anticipation', 'raw_disgust', 'raw_fear', 'raw_joy', 'raw_sadness', 'raw_surprise', 'raw_trust']
    target_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    features = vec.get_feature_names_out()
    features = np.append(features, emo)

    sorted_feature_importance = [None] * len(clf.estimators_)
    for i in range(0, len(clf.estimators_)):
        imp = clf.estimators_[i].coef_[0]
        feature_importance = dict(zip(features, imp))
        sorted_feature_importance[i] = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    for i in range(0, len(clf.estimators_)):
        print(f'\nEmotion: {target_names[i]}')
        j=0
        for feature, importance in sorted_feature_importance[i]:
            if(j<10 or feature in emo):
                print(f'\tFeature: {feature}, Importance: {importance}')
            j += 1
    
    #print(multilabel_confusion_matrix(yy, test),target_names)
    print(classification_report(yy, test, target_names=target_names, zero_division=0.0))
    #print(confusion_matrix(yy, test))   

if __name__ == "__main__":
    main()