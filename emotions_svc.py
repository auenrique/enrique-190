import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

import tokenizer
import emotions

def main():
    data = pd.read_csv('en-annotated.tsv', sep='\t', names=['text', 'label'])
    data.head()

    # Check if there are any missing values.
    data.isnull().sum()

    after_tokenize = tokenizer.tokenize_data(data, True)

    # Split the label column into multiple values
    after_tokenize['labelarr'] = after_tokenize['label'].str.split(',')

    # Extract the first label for each example
    after_tokenize['first_label'] = after_tokenize['labelarr'].apply(lambda x: x[0])

    lex = emotions.build_lexicon()
    emo_features = emotions.get_emotion_features(after_tokenize, lex)

    # print(lex)
    # print (after_tokenize)

    df = pd.DataFrame.from_dict(after_tokenize['label'].values.ravel())
    ef = pd.DataFrame.from_dict(emo_features)

    
    dfnew = pd.DataFrame()
    for i in range(1,9):
        dfnew[str(i)] = df[0].str.contains(str(i)).astype(int)
    dfnew = np.array(dfnew)
    print(dfnew)
    #print(ef.shape())
    # print(df.shape())

    train_dev_X, test_X, train_dev_y, test_y = train_test_split(ef, dfnew, test_size=0.1, stratify=df[0].str.split(',').apply(lambda x: x[0]) , random_state=42)
    train_X, dev_X, train_y, dev_y = train_test_split(train_dev_X, train_dev_y, test_size=0.222, stratify=train_dev_y.str.split(',').apply(lambda x: x[0]), random_state=42)

    clf = OneVsRestClassifier(LinearSVC(class_weight='balanced', C=1.0, dual=False, random_state=42))

    X = train_X
    y = train_y

    ydf = pd.DataFrame()
    for i in range(1,9):
        ydf[str(i)] = y.str.contains(str(i)).astype(int)

    clf.fit(X, ydf)

    XX = test_X
    yy = test_y

    yydf = []
    i = 0
    for labelgrp in yy:
        temp = [0,0,0,0,0,0,0,0]
        for j in range(1,9):
            if str(j) in labelgrp:
                temp[j-1] = 1
            else:
                temp[j-1] = 0
        yydf.append(temp)
        i += 1
    yydf = np.array(yydf)

    test = clf.predict(XX)

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

    for labelgrp in yy:
        for label in labelgrp.split(','):
            #remove whitespace
            label = label.strip()
            total_perlabel[label] += 1
            total += 1
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
    for i in range(len(test)):
        for j in range(1,9):
            if test[i][j-1] == 1:
                if test[i][j-1] == yydf[i][j-1]:
                    correct += 1
                    correct_perlabel[str(j)] += 1
                else:
                    wrong += 1
                    wrong_perlabel[str(j)] += 1
                    fp += 1
                    fp_perlabel[str(j)] += 1
            else:
                if test[i][j-1] != yydf[i][j-1]:
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
    #print(clf.score(XX, yy))
    print(f'Score: {clf.score(XX, yydf)}')

    #get per class f1 scores
    
    #print(f1_score(yy, test, average='micro'))

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

if __name__ == "__main__":
    main()