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
from scipy import stats
from sklearn.metrics import roc_auc_score

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

    #train_model(data, True)

    clf_emo = OneVsRestClassifier(LinearSVC(dual=True, class_weight='balanced', C=0.05, random_state=42))
    clf_noemo = OneVsRestClassifier(LinearSVC(dual=True, class_weight='balanced', C=0.05, random_state=42))

    classification_reports_emo = []
    classification_reports_noemo = []
    auc_scores_emo = []
    auc_scores_noemo = []

    for train_index, test_index in StratifiedKFold(n_splits=10, random_state=42, shuffle=True).split(xd, df.str.replace(' ', '').str.split(',').apply(lambda x: x[0])):
        train_X, test_X = xd[train_index], xd[test_index]
        train_y, test_y = df[train_index], df[test_index]

        train_X = train_X.reset_index(drop=True)
        test_X = test_X.reset_index(drop=True)

        tok_Xtrain = train_X.apply(lambda x: tokenizer.tokenize_nostem_lem(x))
        tok_Xtest = test_X.apply(lambda x: tokenizer.tokenize_nostem_lem(x))

        emo_Xtrain = emotions.get_emotion_features(tok_Xtrain, lex)
        emo_Xtest = emotions.get_emotion_features(tok_Xtest, lex)

        preprocess_Xtrain = tokenizer.tokenize_data(train_X, True)["preprocessed"]
        preprocess_Xtest = tokenizer.tokenize_data(test_X, True)["preprocessed"]

        vec = CountVectorizer(analyzer='word',ngram_range=(1,3))
        mlb = MultiLabelBinarizer() 

        #X = vec.fit_transform(train_X)
        X = vec.fit_transform(preprocess_Xtrain)
        y = mlb.fit_transform(train_y.str.replace(' ', '').str.split(','))

        #XX = vec.transform(test_X)
        XX = vec.transform(preprocess_Xtest)
        yy = mlb.transform(test_y.str.replace(' ', '').str.split(','))

        clf_noemo.fit(X, y)
        #append emotion features to n-gram features
        X_emo = hstack([X, pd.DataFrame(emo_Xtrain)])

        #print(X.shape, y.shape)

        clf_emo.fit(X_emo, y)  

        test_noemo = clf_noemo.predict(XX)

        XX_emo = hstack([XX, pd.DataFrame(emo_Xtest)])
        test_emo = clf_emo.predict(XX_emo)

        target_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        #print(multilabel_confusion_matrix(yy, test),target_names)
        #print(classification_report(yy, test, target_names=target_names))
        classification_reports_emo.append(classification_report(yy, test_emo, target_names=target_names, output_dict=True, zero_division=1.0))
        classification_reports_noemo.append(classification_report(yy, test_noemo, target_names=target_names, output_dict=True, zero_division=1.0))
        auc_scores_emo.append(roc_auc_score(yy, clf_emo.decision_function(XX_emo)))
        auc_scores_noemo.append(roc_auc_score(yy, clf_noemo.decision_function(XX)))

    print('Emotion')
    print('Anger: %.2f' % (sum(report['anger']['f1-score'] for report in classification_reports_emo)/len(classification_reports_emo)))
    print('Anticipation: %.2f' % (sum(report['anticipation']['f1-score'] for report in classification_reports_emo)/len(classification_reports_emo)))
    print('Disgust: %.2f' % (sum(report['disgust']['f1-score'] for report in classification_reports_emo)/len(classification_reports_emo)))
    print('Fear: %.2f' % (sum(report['fear']['f1-score'] for report in classification_reports_emo)/len(classification_reports_emo)))
    print('Joy: %.2f' % (sum(report['joy']['f1-score'] for report in classification_reports_emo)/len(classification_reports_emo)))
    print('Sadness: %.2f' % (sum(report['sadness']['f1-score'] for report in classification_reports_emo)/len(classification_reports_emo)))
    print('Surprise: %.2f' % (sum(report['surprise']['f1-score'] for report in classification_reports_emo)/len(classification_reports_emo)))
    print('Trust: %.2f' % (sum(report['trust']['f1-score'] for report in classification_reports_emo)/len(classification_reports_emo)))
    print('Macro Avg: %.2f' % (sum(report['macro avg']['f1-score'] for report in classification_reports_emo)/len(classification_reports_emo)))

    print('No Emotion')
    print('Anger: %.2f' % (sum(report['anger']['f1-score'] for report in classification_reports_noemo)/len(classification_reports_noemo)))
    print('Anticipation: %.2f' % (sum(report['anticipation']['f1-score'] for report in classification_reports_noemo)/len(classification_reports_noemo)))
    print('Disgust: %.2f' % (sum(report['disgust']['f1-score'] for report in classification_reports_noemo)/len(classification_reports_noemo)))
    print('Fear: %.2f' % (sum(report['fear']['f1-score'] for report in classification_reports_noemo)/len(classification_reports_noemo)))
    print('Joy: %.2f' % (sum(report['joy']['f1-score'] for report in classification_reports_noemo)/len(classification_reports_noemo)))
    print('Sadness: %.2f' % (sum(report['sadness']['f1-score'] for report in classification_reports_noemo)/len(classification_reports_noemo)))
    print('Surprise: %.2f' % (sum(report['surprise']['f1-score'] for report in classification_reports_noemo)/len(classification_reports_noemo)))
    print('Trust: %.2f' % (sum(report['trust']['f1-score'] for report in classification_reports_noemo)/len(classification_reports_noemo)))
    print('Macro Avg: %.2f' % (sum(report['macro avg']['f1-score'] for report in classification_reports_noemo)/len(classification_reports_noemo)))

    # Assuming classification_reports_emo and classification_reports_noemo are lists of dictionaries containing the classification reports
    f1_scores_emo = [report['macro avg']['f1-score'] for report in classification_reports_emo]
    f1_scores_noemo = [report['macro avg']['f1-score'] for report in classification_reports_noemo]
    ang_scores_emo = [report['anger']['f1-score'] for report in classification_reports_emo]
    ang_scores_noemo = [report['anger']['f1-score'] for report in classification_reports_noemo]
    ant_scores_emo = [report['anticipation']['f1-score'] for report in classification_reports_emo]
    ant_scores_noemo = [report['anticipation']['f1-score'] for report in classification_reports_noemo]
    dis_scores_emo = [report['disgust']['f1-score'] for report in classification_reports_emo]
    dis_scores_noemo = [report['disgust']['f1-score'] for report in classification_reports_noemo]
    fear_scores_emo = [report['fear']['f1-score'] for report in classification_reports_emo]
    fear_scores_noemo = [report['fear']['f1-score'] for report in classification_reports_noemo]
    joy_scores_emo = [report['joy']['f1-score'] for report in classification_reports_emo]
    joy_scores_noemo = [report['joy']['f1-score'] for report in classification_reports_noemo]
    sad_scores_emo = [report['sadness']['f1-score'] for report in classification_reports_emo]
    sad_scores_noemo = [report['sadness']['f1-score'] for report in classification_reports_noemo]
    sur_scores_emo = [report['surprise']['f1-score'] for report in classification_reports_emo]
    sur_scores_noemo = [report['surprise']['f1-score'] for report in classification_reports_noemo]
    tru_scores_emo = [report['trust']['f1-score'] for report in classification_reports_emo]
    tru_scores_noemo = [report['trust']['f1-score'] for report in classification_reports_noemo]

    print(f"Emotion\tT-statistic\tP-value")
    t_stat, p_val = stats.ttest_rel(f1_scores_emo, f1_scores_noemo)
    print(f"Macro F1\t{t_stat}\t{p_val}")
    t_stat, p_val = stats.ttest_rel(ang_scores_emo, ang_scores_noemo)
    print(f"Anger\t{t_stat}\t{p_val}")
    t_stat, p_val = stats.ttest_rel(ant_scores_emo, ant_scores_noemo)
    print(f"Anticipation\t{t_stat}\t{p_val}")
    t_stat, p_val = stats.ttest_rel(dis_scores_emo, dis_scores_noemo)
    print(f"Disgust\t{t_stat}\t{p_val}")
    t_stat, p_val = stats.ttest_rel(fear_scores_emo, fear_scores_noemo)
    print(f"Fear\t{t_stat}\t{p_val}")
    t_stat, p_val = stats.ttest_rel(joy_scores_emo, joy_scores_noemo)
    print(f"Joy\t{t_stat}\t{p_val}")
    t_stat, p_val = stats.ttest_rel(sad_scores_emo, sad_scores_noemo)
    print(f"Sadness\t{t_stat}\t{p_val}")
    t_stat, p_val = stats.ttest_rel(sur_scores_emo, sur_scores_noemo)
    print(f"Surprise\t{t_stat}\t{p_val}")
    t_stat, p_val = stats.ttest_rel(tru_scores_emo, tru_scores_noemo)
    print(f"Trust\t{t_stat}\t{p_val}")
    t_stat, p_val = stats.ttest_rel(auc_scores_emo, auc_scores_noemo)
    print(f"AUC\t{t_stat}\t{p_val}")




    #t_stat, p_val = stats.ttest_rel(f1_scores_emo, f1_scores_noemo)
    #todo
    #try other countvectorizers
    #look at other papers that use xed
   

if __name__ == "__main__":
    main()