import os
import numpy as np
import nltk
import pandas as pd

import tokenizer

path = ''

def build_lexicon():
    # Open NRC lexicon
    with open('NRC-Emotion-Intensity-Lexicon-v1.txt', 'r') as f:
        lexicon = f.readlines()

    # Create dictionary to store lexicon
    lex = {}
    for line in lexicon:
        line = line.strip().split('\t')
        word, emotion, value = line[0], line[1], line[2]
        if word not in lex.keys():
            lex[word] = {}
        lex[word][emotion] = float(value)

    return lex

#def build_lexicon_from_text(text, lex):

def search_lexicon(word, lex):
    if word in lex.keys():
        return lex[word]
    else:
        return None
    
def build_raw_score(text, lex):
    raw = {}
    raw['raw_anger'] = 0
    raw['raw_anticipation'] = 0
    raw['raw_disgust'] = 0
    raw['raw_fear'] = 0
    raw['raw_joy'] = 0
    raw['raw_sadness'] = 0
    raw['raw_surprise'] = 0
    raw['raw_trust'] = 0

    for word in text:
        search_lexicon(word, lex)
        if(search_lexicon(word, lex) != None):
            for emotion in search_lexicon(word, lex):
                raw['raw_'+emotion] += search_lexicon(word, lex)[emotion]
    return raw

def get_emotion_words (text, lex):
    emotion_words = []
    for word in text:
        is_emotional = False
        if(search_lexicon(word, lex) != None):
            for emotion in search_lexicon(word, lex):
                if(search_lexicon(word, lex)[emotion] != 0 and is_emotional == False):
                    #add word to list of emotional words
                    emotion_words.append(word)
                    is_emotional = True
    return emotion_words

def get_emotionality (text, lex):
    emotion_words = get_emotion_words(text, lex)
    emotion_word_cnt = len(emotion_words)
    total_words = len(text)
    if(total_words == 0):
        total_words = 1
    emotionality = emotion_word_cnt/total_words
    return emotionality

def get_avg_emotion (text, lex):
    raw = build_raw_score(text, lex)
    emotion_words = get_emotion_words(text, lex)
    emotion_words_cnt = len(emotion_words)
    avg = {}
    if(emotion_words_cnt == 0):
        emotion_words_cnt = 1
    avg['avg_anger'] = raw['raw_anger']/emotion_words_cnt
    avg['avg_anticipation'] = raw['raw_anticipation']/emotion_words_cnt
    avg['avg_disgust'] = raw['raw_disgust']/emotion_words_cnt
    avg['avg_fear'] = raw['raw_fear']/emotion_words_cnt
    avg['avg_joy'] = raw['raw_joy']/emotion_words_cnt
    avg['avg_sadness'] = raw['raw_sadness']/emotion_words_cnt
    avg['avg_surprise'] = raw['raw_surprise']/emotion_words_cnt
    avg['avg_trust'] = raw['raw_trust']/emotion_words_cnt

    return avg

def get_pct_emotion(text,lex):
    raw = build_raw_score(text, lex)
    raw_sum = raw['raw_anger'] + raw['raw_anticipation'] + raw['raw_disgust'] + raw['raw_fear'] + raw['raw_joy'] + raw['raw_sadness'] + raw['raw_surprise'] + raw['raw_trust']
    pct = {}
    if(raw_sum == 0):
        raw_sum = 1
    pct['pct_anger'] = raw['raw_anger']/raw_sum
    pct['pct_anticipation'] = raw['raw_anticipation']/raw_sum
    pct['pct_disgust'] = raw['raw_disgust']/raw_sum
    pct['pct_fear'] = raw['raw_fear']/raw_sum
    pct['pct_joy'] = raw['raw_joy']/raw_sum
    pct['pct_sadness'] = raw['raw_sadness']/raw_sum
    pct['pct_surprise'] = raw['raw_surprise']/raw_sum
    pct['pct_trust'] = raw['raw_trust']/raw_sum

    return pct

def get_pct_sentiment(text,lex):
    raw = build_raw_score(text, lex)
    raw_sum = raw['raw_negative'] + raw['raw_positive']
    pct = {}
    if(raw_sum == 0):
        raw_sum = 1
    pct['pct_negative'] = raw['raw_negative']/raw_sum
    pct['pct_positive'] = raw['raw_positive']/raw_sum

    return pct

def get_emotion_pct(data):
    print(data)
    # for thing in data:
    #     print(data[thing])

def get_emoword_cnt(data,lex):
    emotion_words = data.apply(lambda x: get_emotion_words(x, lex))
    for i in range(len(emotion_words)):
        emotion_words[i] = len(emotion_words[i])
    return emotion_words

def get_emotion_features (data, lex):
    emotion_features = dict()
    #copy data to emotion_features

    #add emotion features to emotion_features
    #print('getting raw scores...')
    raw_scores = data.apply(lambda x: build_raw_score(x, lex))
    for emotion in raw_scores[0]:
        emotion_features[emotion] = raw_scores.apply(lambda x: x[emotion])
    #emotion_features = pd.concat([emotion_features, raw_scores.apply(pd.Series)], axis=1)
    #print(raw_scores)
    #raw_scores.index = ['raw_anger', 'raw_anticipation', 'raw_disgust', 'raw_fear', 'raw_joy', 'raw_negative', 'raw_positive', 'raw_sadness', 'raw_surprise', 'raw_trust']
    # print('getting emotion word count...')
    # emotion_words = emotion_features['preprocessed'].apply(lambda x: get_emotion_words(x, lex))
    # emotion_features['emotion_words'] = emotion_words

    # print('getting emotionality...')
    # emotionality = data.apply(lambda x: get_emotionality(x, lex))
    # emotion_features['emotionality'] = emotionality

    # print('getting avg emotion scores...')
    # avg_emotion_scores = data.apply(lambda x: get_avg_emotion(x, lex))
    # for emotion in avg_emotion_scores[0]:
    #     emotion_features[emotion] = avg_emotion_scores.apply(lambda x: x[emotion])

    # print('getting emotion percentages...')
    # pct_emotion_scores = data.apply(lambda x: get_pct_emotion(x, lex))
    # for emotion in pct_emotion_scores[0]:
    #     emotion_features[emotion] = pct_emotion_scores.apply(lambda x: x[emotion])

    # print('getting sentiment percentages...')
    # pct_sentiment_scores = emotion_features['preprocessed'].apply(lambda x: get_pct_sentiment(x, lex))
    # for sentiment in pct_sentiment_scores[0]:
    #     emotion_features[sentiment] = pct_sentiment_scores.apply(lambda x: x[sentiment])

    #print(emotion_features)
    # print(raw_scores[3])
    # print(emotion_word_count[3])
    # print(emotionality[3])
    # print(avg_emotion_scores[3])
    # print(pct_emotion_scores[3])
    return emotion_features
    

def main():
    data  =  pd.read_csv('devp_partial.tsv', sep='\t')
    data.head()

    # Check if there are any missing values.
    data.isnull().sum()

    lex = build_lexicon()

    after_tokenize = tokenizer.tokenize_data(data, True)

    emo_features = get_emotion_features(after_tokenize, lex)
    #print(emo_features)
    
    df = pd.DataFrame.from_dict(emo_features) 
    df.to_csv (r'emotest.csv', index = False, header=True)
    
    #print(data["preprocessed"][0][0])
    # df = pd.DataFrame.from_dict(after_tokenize) 
    # df.to_csv (r'gen.csv', index = False, header=True)

if __name__ == "__main__":
    main()