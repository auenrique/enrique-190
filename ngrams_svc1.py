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
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import tokenizer
import emotions

def get_dataset_stats(data):
    sentcnt = 0
    labelcnt = 0
    onecnt = 0
    twocnt = 0
    threecnt = 0
    fourpluscnt = 0
    angercnt = 0
    anticipationcnt = 0
    disgustcnt = 0
    fearcnt = 0
    joycnt = 0
    sadnesscnt = 0
    surprisecnt = 0
    trustcnt = 0
    for x in data:
        x_arr = x.replace(' ','').split(',')
        labelcnt += len(x_arr)
        sentcnt += 1
        match len(x_arr):
            case 1:
                onecnt += 1
            case 2:
                twocnt += 1
            case 3:
                threecnt += 1
            case _:
                fourpluscnt += 1
        for i in x_arr:
            if i == '1':
                angercnt += 1
            elif i == '2':
                anticipationcnt += 1
            elif i == '3':
                disgustcnt += 1
            elif i == '4':
                fearcnt += 1
            elif i == '5':
                joycnt += 1
            elif i == '6':
                sadnesscnt += 1
            elif i == '7':
                surprisecnt += 1
            elif i == '8':
                trustcnt += 1
    print(f'Number of labels: {labelcnt}')
    print(f'Number of sentences: {sentcnt}')
    print('Number of labels per sentence:')
    print(f'\t1: {onecnt/labelcnt*100}%')
    print(f'\t2: {twocnt/labelcnt*100}%')
    print(f'\t3: {threecnt/labelcnt*100}%')
    print(f'\t4+: {fourpluscnt/labelcnt*100}%')
    print('Number of each emotion:')
    print(f'\tanger: {angercnt/labelcnt*100}%')
    print(f'\tanticipation: {anticipationcnt/labelcnt*100}%')
    print(f'\tdisgust: {disgustcnt/labelcnt*100}%')
    print(f'\tfear: {fearcnt/labelcnt*100}%')
    print(f'\tjoy: {joycnt/labelcnt*100}%')
    print(f'\tsadness: {sadnesscnt/labelcnt*100}%')
    print(f'\tsurprise: {surprisecnt/labelcnt*100}%')
    print(f'\ttrust: {trustcnt/labelcnt*100}%')     


def get_emocnt_pct(data):
    lex = emotions.build_lexicon()
    emowdcnt = emotions.get_emoword_cnt(data.apply(lambda x: tokenizer.tokenize_nostem(x)), lex)
    total = 0
    cnt = 0
    for i in range(0, len(emowdcnt)):
        total += 1
        if emowdcnt[i] > 0:
            cnt += 1
    return cnt/total

def get_intensity_per_emo(data):
    lex = emotions.build_lexicon()
    emoval = pd.DataFrame.from_dict(emotions.get_emotion_features(data['text'].apply(lambda x: tokenizer.tokenize_nostem(x)), lex))
    labels = data['label'].str.replace(' ', '').str.split(',')
    index = 0
    # get average intensity for each label
    avg = [[0 for _ in range(8)] for _ in range(8)]
    avgcnt = [0] * 8
    for label in labels:
        for i in range(0, len(label)):
            emo = int(label[i])-1
            avg[emo][0] += emoval.iloc[index]['raw_anger']
            avg[emo][1] += emoval.iloc[index]['raw_anticipation']
            avg[emo][2] += emoval.iloc[index]['raw_disgust']
            avg[emo][3] += emoval.iloc[index]['raw_fear']
            avg[emo][4] += emoval.iloc[index]['raw_joy']
            avg[emo][5] += emoval.iloc[index]['raw_sadness']
            avg[emo][6] += emoval.iloc[index]['raw_surprise']
            avg[emo][7] += emoval.iloc[index]['raw_trust']
            avgcnt[emo] += 1
        index += 1
    for i in range(0, 8):
        for j in range(0, 8):
            avg[i][j] /= avgcnt[i]
        print(avg[i])
        print(avgcnt[i])

def train_model(train_X, train_y, test_X, test_y, use_intensity):
    vec = CountVectorizer(analyzer='word',ngram_range=(1,3))
    mlb = MultiLabelBinarizer()

    preprocess_Xtrain = tokenizer.tokenize_data(train_X, True)["preprocessed"]
    preprocess_Xtest = tokenizer.tokenize_data(test_X, True)["preprocessed"]

    X = vec.fit_transform(preprocess_Xtrain)
    y = mlb.fit_transform(train_y.str.replace(' ', '').str.split(','))

    XX = vec.transform(preprocess_Xtest)
    yy = mlb.transform(test_y.str.replace(' ', '').str.split(','))
    features = vec.get_feature_names_out()

    if use_intensity:
        tok_Xtrain = train_X.apply(lambda x: tokenizer.tokenize_nostem(x))
        tok_Xtest = test_X.apply(lambda x: tokenizer.tokenize_nostem(x))

        lex = emotions.build_lexicon()
        emo_Xtrain = emotions.get_emotion_features(tok_Xtrain, lex)
        emo_Xtest = emotions.get_emotion_features(tok_Xtest, lex)
        X = hstack([X, pd.DataFrame(emo_Xtrain)])
        XX = hstack([XX, pd.DataFrame(emo_Xtest)])
        emo = ['raw_anger', 'raw_anticipation', 'raw_disgust', 'raw_fear', 'raw_joy', 'raw_sadness', 'raw_surprise', 'raw_trust']
        features = np.append(features, emo)

    clf = OneVsRestClassifier(LinearSVC(dual=True, class_weight='balanced', C=0.01, random_state=42))
    clf.fit(X, y)    

    test = clf.predict(XX)

    return get_clf_metrics(clf, yy, test, features)

def get_clf_metrics(clf, y_true, y_pred, features):
    target_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    emo = ['raw_anger', 'raw_anticipation', 'raw_disgust', 'raw_fear', 'raw_joy', 'raw_sadness', 'raw_surprise', 'raw_trust']

    sorted_feature_importance = [None] * len(clf.estimators_)
    for i in range(0, len(clf.estimators_)):
        imp = clf.estimators_[i].coef_[0]
        feature_importance = dict(zip(features, imp))
        sorted_feature_importance[i] = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    print(features)
    emoint_dict = {
        'anger': {},
        'anticipation': {},
        'disgust': {},
        'fear': {},
        'joy': {},
        'sadness': {},
        'surprise': {},
        'trust': {}
    }
    for i in range(0, len(clf.estimators_)):
        print(f'\nEmotion: {target_names[i]}')
        ngram = []
        emo_int = []
        
        j=0
        for feature, importance in sorted_feature_importance[i]:
            if(feature in emo):
                emo_int.append(f'\tFeature: {feature}, Importance: {importance}')
                match i:
                    case 0:
                        emoint_dict['anger'][feature] = importance
                    case 1:
                        emoint_dict['anticipation'][feature] = importance
                    case 2:
                        emoint_dict['disgust'][feature] = importance
                    case 3:
                        emoint_dict['fear'][feature] = importance
                    case 4:
                        emoint_dict['joy'][feature] = importance
                    case 5:
                        emoint_dict['sadness'][feature] = importance
                    case 6:
                        emoint_dict['surprise'][feature] = importance
                    case 7:
                        emoint_dict['trust'][feature] = importance
            if(len(ngram)<10 and feature not in emo):
                ngram.append(f'\tFeature: {feature}, Importance: {importance}')
        for n in ngram:
            print(n)
        for e in emo_int:
            print(e)
    #if emoint_dict is filled, call visualize_importance(emoint_dict)
    if emoint_dict['anger']:
        visualize_importance(emoint_dict)

    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0.0))
    #print(roc_auc_score(y_true, y_pred, average=None))
    print(f'AUC: {roc_auc_score(y_true, y_pred, average=None)}')
    print(f'F1: {f1_score(y_true, y_pred, average="samples")}')

    return(classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0.0))

def visualize_importance(emos):
    #create heatmap
    fig, ax = plt.subplots()
    #arrange values in a list of lists
    arr = [[0 for _ in range(8)] for _ in range(8)]
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    emo_int = ['anger_int', 'anticipation_int', 'disgust_int', 'fear_int', 'joy_int', 'sadness_int', 'surprise_int', 'trust_int']
    for i, emotion in enumerate(emotions):
        for j, (feature, importance) in enumerate(emos[emotion].items()):
            match feature:
                case 'raw_anger':
                    arr[i][0] = importance
                case 'raw_anticipation':
                    arr[i][1] = importance
                case 'raw_disgust':
                    arr[i][2] = importance
                case 'raw_fear':
                    arr[i][3] = importance
                case 'raw_joy':
                    arr[i][4] = importance
                case 'raw_sadness':
                    arr[i][5] = importance
                case 'raw_surprise':
                    arr[i][6] = importance
                case 'raw_trust':
                    arr[i][7] = importance

    plt.subplots_adjust(bottom=0.271, top=0.895, right=0.874, left=0.124)
    im = ax.imshow(arr)
    ax.set_xticks(np.arange(len(emos['anger'])))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(emo_int)
    ax.set_yticklabels(emotions)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(8):
        for j in range(len(emos['anger'])):
            text = ax.text(j, i, round(arr[i][j], 2), ha="center", va="center", color="w")
    #add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Feature Importance', rotation=-90, va="bottom")
    ax.set_title('Emotion Intensity Importance')
    #add title to x and y axis
    ax.set_xlabel('Emotion Intensity Values')
    ax.set_ylabel('Emotion Labels')
    plt.show()

def main():
    data = pd.read_csv('en-annotated.tsv', sep='\t', names=['text', 'label'])
    data.head()

    # Check if there are any missing values.
    print(data.isnull().sum())
    
    after_tokenize = tokenizer.tokenize_fromdf(data, True)

    dd = pd.DataFrame.from_dict(after_tokenize)
    dd.to_csv ('help.tsv', sep='\t', index=False)

    df = data['label']

    xd = data['text']
    
    #print(get_emocnt_pct(xd))
    #get_intensity_per_emo(data)
    #get_dataset_stats(df)

    train_dev_X, test_X, train_dev_y, test_y = train_test_split(xd, df, test_size=0.1, stratify=df.str.replace(' ', '').str.split(',').apply(lambda x: x[0]), random_state=42)
    train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, stratify=train_dev_y.str.split(',').apply(lambda x: x[0]), random_state=42)

    # train_dev_X, test_X, train_dev_y, test_y = train_test_split(xd, df[0], test_size=0.1, random_state=42)
    # train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, random_state=42)
    # ddf = pd.DataFrame(df[0].str.split(',').apply(lambda x: x[0]))
    
    # train_dev_X, test_X, train_dev_y, test_y = train_test_split(xd, ddf[0], test_size=0.1, stratify=ddf[0], random_state=42)
    # train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, stratify=train_dev_y, random_state=42)

    #clf = LinearSVC(class_weight='balanced', C=0.05, random_state=42)
    # test_X = xd
    # test_y = df

    train_X = train_dev_X.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)

    train_y = train_dev_y.reset_index(drop=True)

    cr_int = train_model(train_X, train_y, test_X, test_y, True)
    cr_noint = train_model(train_X, train_y, test_X, test_y, False)

if __name__ == "__main__":
    main()