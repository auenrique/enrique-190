# contains generic preprocessing functions

import numpy as np
import nltk
import string
import re
import pandas as pd

def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    #remove numbers
    no_punct = re.sub(r'\d+', '', no_punct)
    no_punct = no_punct.replace('\n', ' ')
    no_punct = no_punct.strip()
    return no_punct

def tokenize(text):
    tokens = re.split(r'\W+', text)
    return tokens

def remove_stopwords(stopword, tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

def stemmer(porter, tokenized_text):
    text = [porter.stem(word) for word in tokenized_text]
    return text

def lemmatizer_func(lemmatizer, tokenized_text):
    text = [lemmatizer.lemmatize(word) for word in tokenized_text]
    return text

def stemmer_func(tokenized_text):
    stemmer = nltk.SnowballStemmer('english')
    text = [stemmer.stem(word) for word in tokenized_text]
    return text


def tokenize_stream(text):
    text = text.lower()
    text = remove_punctuation(text)    
    text_arr = tokenize(text)
    text_arr = stemmer_func(text_arr)
    
    return text_arr

def preprocess_stream(tokenized_text):
    #convert text array back to string
    text = ' '.join(tokenized_text)
    return text

def tokenize_nostem(text):
    text = text.lower()
    text = remove_punctuation(text)
    text_arr = tokenize(text)
    return text_arr

def tokenize_nostem_lem(text):
    text = text.lower()
    text = remove_punctuation(text)
    text_arr = tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    text_arr = lemmatizer_func(lemmatizer, text_arr)
    return text_arr

def tokenize_fromdf(data,fast):
    tokenized = dict()
    if(fast):
        tokenized = dict()
        tokenized["text"] = data["text"]
        tokenized["tokenized"] = tokenized["text"].apply(lambda x: tokenize_stream(x))
        tokenized["tok_nostem"] = tokenized["text"].apply(lambda x: tokenize_nostem(x))
        tokenized["preprocessed"] = tokenized["tokenized"].apply(lambda x: preprocess_stream(x))
        tokenized["label"] = data["label"]
    #verbose
    else:
        tokenized["text"] = data["text"]
        print('removing special characters...')
        tokenized["text_nopunct"] = tokenized["text"].apply(lambda x: remove_punctuation(x))
        print('downcasing...')
        tokenized["text_downcase"] = tokenized["text_nopunct"].apply(lambda x: x.lower())
        print('tokenizing...')
        tokenized["tokenized"] = tokenized["text_downcase"].apply(lambda x: tokenize(x))

        print('removing stopwords...')
        stopword = nltk.corpus.stopwords.words('english')
        tokenized["nostop"] = tokenized["tokenized"].apply(lambda x: remove_stopwords(stopword, x))

        print('lemmatizing...')
        lemmatizer = nltk.WordNetLemmatizer()
        #tokenize_alldata["stemmed"] = tokenize_alldata["nostop"].apply(lambda x: stemmer(porter, x))
        tokenized["lemmatized"] = tokenized["nostop"].apply(lambda x: lemmatizer_func(lemmatizer, x))

        tokenized["label"] = data["label"]


    return tokenized

#returns data for all steps of tokenization
def tokenize_data(data,fast):
    tokenized = dict()
    if(fast):
        tokenized = dict()
        tokenized["text"] = data
        stopword = nltk.corpus.stopwords.words('english')
        lemmatizer = nltk.WordNetLemmatizer()
        tokenized["tokenized"] = tokenized["text"].apply(lambda x: tokenize_stream(x))
        tokenized["tok_nostem"] = tokenized["text"].apply(lambda x: tokenize_nostem(x))
        tokenized["preprocessed"] = tokenized["tokenized"].apply(lambda x: preprocess_stream(x))
    #verbose
    else:
        tokenized["text"] = data["text"]
        print('removing special characters...')
        tokenized["text_nopunct"] = tokenized["text"].apply(lambda x: remove_punctuation(x))
        print('downcasing...')
        tokenized["text_downcase"] = tokenized["text_nopunct"].apply(lambda x: x.lower())
        print('tokenizing...')
        tokenized["tokenized"] = tokenized["text_downcase"].apply(lambda x: tokenize(x))

        print('removing stopwords...')
        stopword = nltk.corpus.stopwords.words('english')
        tokenized["nostop"] = tokenized["tokenized"].apply(lambda x: remove_stopwords(stopword, x))

        print('lemmatizing...')
        lemmatizer = nltk.WordNetLemmatizer()
        #tokenize_alldata["stemmed"] = tokenize_alldata["nostop"].apply(lambda x: stemmer(porter, x))
        tokenized["lemmatized"] = tokenized["nostop"].apply(lambda x: lemmatizer_func(lemmatizer, x))

    return tokenized

def tokener():
    data  =  pd.read_csv('dev.tsv', sep='\t')
    data.head()

    # Check if there are any missing values.
    data.isnull().sum()

    after_tokenize = tokenize_data(data, False)
    #print(after_tokenize["lemmatized"])
    df = pd.DataFrame.from_dict(after_tokenize) 
    df.to_csv ('devp.tsv', sep='\t', index=False)

    preprocessed_only = tokenize_data(data, True)
    df2 = pd.DataFrame.from_dict(preprocessed_only)
    df2.to_csv ('devp_partial.tsv', sep='\t', index=False)

def main():
    tokener()

if __name__ == "__main__":
    main()