import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import numpy as np

import tokenizer

def to_list(df, attribute):
    # Select the normalised transcript column 
    df_transcription = df[[attribute]]
    df_transcription = df_transcription.dropna(subset=[attribute])
    # To convert the attribute into list format, but it has inner list. So it cannot put into the CountVectoriser
    unflat_list_transcription = df_transcription.values.tolist()
    # Let's use back the function defined above, "flat_list", to flatten the list
    flat_list_transcription = [item for sublist in unflat_list_transcription for item in sublist]
    return flat_list_transcription

def generate_n_gram_features(flat_list_transcription):
    n_gram_features ={'unigram':(1,1),'unigram_bigram':(1,2),'bigram':(2,2),'bigram_trigram':(2,3),'trigram':(3,3), 'unigram_bigram_trigram':(1,3)}
    temp=[]
    for key, values in n_gram_features.items(): 
        vectorizer = CountVectorizer(ngram_range=values)
        vectorizer.fit(flat_list_transcription)
        temp.append(vectorizer.transform(flat_list_transcription))
    return temp

def main():
    data  =  pd.read_csv('devp_partial.tsv', sep='\t')
    data.head()

    # Check if there are any missing values.
    data.isnull().sum()

    after_tokenize = tokenizer.tokenize_data(data, True)

    # total_word_count = data['text'].str.split().str.len().sum()
    # total_word_count_normalised = data['preprocessed'].str.split().str.len().sum()
    # print(f'The word count of transcription after normalised is: {int(total_word_count_normalised)}')
    # print(f'{round((total_word_count - total_word_count_normalised)/total_word_count*100, 2)}% less word')
    print(after_tokenize['preprocessed'].count())
    #flat_list_transcription = to_list(after_tokenize, 'preprocessed')
    
    temp = generate_n_gram_features(after_tokenize['preprocessed'])
    # for string in data['preprocessed']:
    #     print(string)
    #     if string == :
    #         print('yes')
    
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

    n_gram_df = pd.DataFrame({'N-Gram Feature Vector':feature_vector, 'Data Dimension':feature_vector_shape})
    print(n_gram_df)

    clf = LinearSVC()

    X = temp[5]
    y = data['label']

    clf.fit(X, y)

    print(clf.score(X, y))

    data2  =  pd.read_csv('test.tsv', sep='\t')
    data2.head()
    data2.isnull().sum()
    after_tokenize2 = tokenizer.tokenize_data(data2, True)
    temp2 = generate_n_gram_features(after_tokenize2['preprocessed'])

    print(clf.score(temp2[5], data2['label']))  


if __name__ == "__main__":
    main()