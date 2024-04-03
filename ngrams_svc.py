import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix

import tokenizer
import emotions

def generate_n_gram_features(flat_list_transcription):
    n_gram_features ={'unigram':(1,1),'unigram_bigram':(1,2),'bigram':(2,2),'bigram_trigram':(2,3),'trigram':(3,3), 'unigram_bigram_trigram':(1,3)}
    temp=[]
    for key, values in n_gram_features.items(): 
        #vectorizer = CountVectorizer(ngram_range=values)
        vectorizer = CountVectorizer(ngram_range=values, strip_accents='unicode')

        vectorizer.fit(flat_list_transcription)
        # print(vectorizer.get_feature_names_out())
        temp.append(vectorizer.transform(flat_list_transcription))
    return temp

#performs multi-label encoding of labels
#example: 1,3,7 -> [1,0,1,0,0,0,1,0]
def to_array(data):
    arr = []
    for i in data:
        temp = [0,0,0,0,0,0,0,0]
        for j in range(1,9):
            if str(j) in i:
                temp[j-1] = 1
            else:
                temp[j-1] = 0
        arr.append(temp)

    return np.array(arr)

def generate_metrics(y_true, y_pred):
    correct = 0
    wrong = 0
    fp = 0
    fn = 0
    total = 0

    correct_perlabel = {}
    wrong_perlabel = {}
    fp_perlabel = {}
    fn_perlabel = {}
    total_perlabel = {}
    for i in range(1,9):
        correct_perlabel[str(i)] = 0
        wrong_perlabel[str(i)] = 0
        fp_perlabel[str(i)] = 0
        fn_perlabel[str(i)] = 0
        total_perlabel[str(i)] = 0

    for labelgrp in y_true:
        for i in range(1,9):
            if labelgrp[i-1] == 1:
                total_perlabel[str(i)] += 1
                total += 1

    for i in range(len(y_pred)):
        for j in range(1,9):
            if y_pred[i][j-1] == 1:
                if y_pred[i][j-1] == y_true[i][j-1]:
                    correct += 1
                    correct_perlabel[str(j)] += 1
                else:
                    wrong += 1
                    wrong_perlabel[str(j)] += 1
                    fp += 1
                    fp_perlabel[str(j)] += 1
            else:
                if y_pred[i][j-1] != y_true[i][j-1]:
                    fn += 1
                    fn_perlabel[str(j)] += 1
                    wrong += 1
                    wrong_perlabel[str(j)] += 1

    print(f'Correct: {correct}')
    print(f'Wrong: {wrong}')
    print(f'Total: {total}')
    print(f'FP: {fp}')
    print(f'FN: {fn}')
    print(f'Precision: {correct/(correct+fp)}')
    print(f'Recall: {correct/(correct+fn)}')
    print(f'F1: {2*((correct/(correct+fp))*(correct/(correct+fn)))/((correct/(correct+fp))+(correct/(correct+fn)))}')
    print(f'Accuracy: {(correct-wrong)/total}')
    #fr = f1_score(y_true, y_pred, average='weighted')
    print(f'Calc F1: {f1_score(y_true, y_pred, average="weighted")}')
    

    print('correct per label')
    print(correct_perlabel)

    print('wrong per label')
    print(wrong_perlabel)

    print('total per label')
    print(total_perlabel)

    print('fp per label')  
    print(fp_perlabel)

    print('fn per label')
    print(fn_perlabel)

    print('f1 per label')
    f1_perlabel = {}
    for i in range(1,9):
        f1_perlabel[str(i)] = 2*((correct_perlabel[str(i)]/(correct_perlabel[str(i)]+fp_perlabel[str(i)]))*(correct_perlabel[str(i)]/(correct_perlabel[str(i)]+fn_perlabel[str(i)]))/((correct_perlabel[str(i)]/(correct_perlabel[str(i)]+fp_perlabel[str(i)]))+(correct_perlabel[str(i)]/(correct_perlabel[str(i)]+fn_perlabel[str(i)]))))

    print(f1_perlabel)

def main():
    data = pd.read_csv('en-annotated.tsv', sep='\t', names=['text', 'label'])
    data.head()

    # Check if there are any missing values.
    print(data.isnull().sum())
    
    after_tokenize = tokenizer.tokenize_data(data, True)

    dd = pd.DataFrame.from_dict(after_tokenize)
    #print(dd.shape)
    #drop columns where preprocessed is empty
    #dd = dd[dd['preprocessed'].map(lambda d: len(d)) > 0]
    #print(dd.shape)
    
    temp = generate_n_gram_features(dd['preprocessed'])
    
    dataframes = {'unigram':temp[0], 
              'unigram_bigram':temp[1], 
              'bigram':temp[2], 
              'bigram_trigram':temp[3], 
              'trigram':temp[4],
              'unigram_bigram_trigram':temp[5]}
    feature_vector = [] ; feature_vector_shape = []
    for key in dataframes:
        feature_vector.append(key)
        feature_vector_shape.append(dataframes[key].shape)

    # n_gram_df = pd.DataFrame({'N-Gram Feature Vector':feature_vector, 'Data Dimension':feature_vector_shape})
    # print(n_gram_df)
    
    df = pd.DataFrame.from_dict(dd['label'].values.ravel())
    #print(df[0])
    # print(df)
    # labeldf = pd.DataFrame()
    # for i in range(1,9):
    #     labeldf[str(i)] = df[0].str.contains(str(i)).astype(int)

    #print(labeldf)

    lex = emotions.build_lexicon()
    emo_features = emotions.get_emotion_features(after_tokenize, lex)

    ef = pd.DataFrame.from_dict(emo_features)
    #xd = pd.DataFrame(dataframes['unigram_bigram_trigram'].toarray())

    # ad = pd.DataFrame(dataframes['unigram'].toarray())
    # bd = pd.DataFrame(dataframes['bigram'].toarray())
    # cd = pd.DataFrame(dataframes['trigram'].toarray())
    # dd = pd.concat([ad, bd, cd], axis=1)
    #dd = np.concatenate((dataframes['unigram'].toarray(), dataframes['bigram'].toarray(), dataframes['trigram'].toarray()), axis=1)
    #merge emo_features and n-gram features
    # xd = pd.concat([ef, xd], axis=1)
    #xd = np.array(dataframes['unigram_bigram_trigram'].toarray())

    #tf = TfidfTransformer()
    #dataframes['unigram_bigram_trigram'] = tf.fit_transform(dataframes['unigram_bigram_trigram'])

    train_dev_X, test_X, train_dev_y, test_y = train_test_split(dataframes['unigram_bigram_trigram'], df[0], test_size=0.1, stratify=df[0].str.split(',').apply(lambda x: x[0]), random_state=42)
    train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.16666, stratify=train_dev_y.str.split(',').apply(lambda x: x[0]), random_state=42)

    # df = pd.DataFrame.from_dict(after_tokenize['first_label'].values.ravel())
    
    # train_dev_X, test_X, train_dev_y, test_y = train_test_split(dataframes['unigram_bigram_trigram'], df, test_size=0.1, stratify=df[0], random_state=42)
    # train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, stratify=train_dev_y[0], random_state=42)

    #clf = LinearSVC(penalty='l1', dual=False, class_weight='balanced', random_state=42)
    clf = OneVsRestClassifier(LinearSVC(class_weight='balanced', C=0.05, random_state=42))

    #print(train_y.str.split(','))
    X = train_X
    y = to_array(train_y)

    # ydf = pd.DataFrame()
    # for i in range(1,9):
    #     ydf[str(i)] = y.str.contains(str(i)).astype(int)

    #ydf = to_array(y)

    clf.fit(X, y)

    XX = test_X
    yy = to_array(test_y)

    # yydf = []
    # i = 0
    # for labelgrp in yy:
    #     temp = [0,0,0,0,0,0,0,0]
    #     for j in range(1,9):
    #         if str(j) in labelgrp:
    #             temp[j-1] = 1
    #         else:
    #             temp[j-1] = 0
    #     yydf.append(temp)
    #     i += 1
    # yydf = np.array(yydf)
    #yydf = to_array(yy)

    test = clf.predict(XX)

    #print('Score: ', clf.score(XX, yy))
    #generate_metrics(yy, test)

    print(set(yy.ravel()) - set(test.ravel()))

    target_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    #print(multilabel_confusion_matrix(yy, test),target_names)
    print(classification_report(yy, test, target_names=target_names))

    #todo
    # improve model accuracy
    # tf-idf?


    i = 0
    # for labelgrp in yy:
    #     if(i>=0):
    #         predictcount = 0
    #         # print(f'label: {labelgrp}')
    #         # print(f'prediction: {test[i]}')
    #         for label in labelgrp.split(','):
    #             label = label.strip()
    #             if test[i][int(label)-1] == 1:
    #                 correct += 1
    #                 correct_perlabel[label] += 1
    #             else:
    #                 wrong += 1
    #                 wrong_perlabel[label] += 1
    #                 fp += 1
    #                 fp_perlabel[label] += 1
    #             predictcount += 1
    #         #get number of 1s in prediction
    #         n = 0    
    #         for j in range(1,9):
    #             if test[i][j-1] == 1:
    #                 n += 1
    #         if predictcount < n:
    #             wrong += n - predictcount
    #             for j in range(1,9):
    #                 if test[i][j-1] == 1:
    #                     if str(j) not in labelgrp:
    #                         fn += 1
    #                         fn_perlabel[str(j)] += 1
    #         i += 1
   

if __name__ == "__main__":
    main()