import numpy as np
import nltk
import pandas as pd

import tokenizer

def generate_N_grams(text,ngram):
  words=[word for word in text.split(" ")]  
  #print("Sentence after removing stopwords:",words)
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans

def main():
    data  =  pd.read_csv('devp_partial.tsv', sep='\t')
    data.head()

    # Check if there are any missing values.
    data.isnull().sum()

    after_tokenize = tokenizer.tokenize_data(data, True)

    print(after_tokenize['preprocessed'].count())

    after_tokenize['1gram'] = after_tokenize['preprocessed'].apply(lambda x: generate_N_grams(x,1))
    after_tokenize['2gram'] = after_tokenize['preprocessed'].apply(lambda x: generate_N_grams(x,2))
    after_tokenize['3gram'] = after_tokenize['preprocessed'].apply(lambda x: generate_N_grams(x,3))

    df = pd.DataFrame.from_dict(after_tokenize)
    print(df['preprocessed'].count())
    # df.to_csv ('devp_ngrams.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main()