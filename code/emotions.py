# contains functions for emotion feature extraction using the NRC Emotion Intensity Lexicon
def build_lexicon():
    # Open NRC lexicon
    with open('data/NRC-Emotion-Intensity-Lexicon-v1.txt', 'r') as f:
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

# provides an overview of the lexicon (table 1)
def show_lexicon_stats(lex):
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    emotion_counts = {emotion: 0 for emotion in emotions}
    for word in lex.keys():
        for emotion in lex[word].keys():
            emotion_counts[emotion] += 1
    #get number of words in lexicon
    word_count = len(lex.keys())
    print('Number of words in lexicon: ', word_count)
    #get number of emotions in lexicon
    emotion_count = len(emotions)
    #get sum of emotion counts
    emotion_sum = sum(emotion_counts.values())
    print('Number of unique intensity values: ', emotion_sum)
    #get proportion
    for emotion in emotion_counts:
        emotion_counts[emotion] = emotion_counts[emotion]/emotion_sum
    print(emotion_counts)

# search lexicon for a word, returns intensity values for each emotion
def search_lexicon(word, lex):
    if word in lex.keys():
        return lex[word]
    else:
        return None

# build raw score by adding intensity values of each word in text
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

#unused in study
#return list of words in text that appear in lexicon
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

#unused in study
#return % of words in text that appear in lexicon
def get_emotionality (text, lex):
    emotion_words = get_emotion_words(text, lex)
    emotion_word_cnt = len(emotion_words)
    total_words = len(text)
    if(total_words == 0):
        total_words = 1
    emotionality = emotion_word_cnt/total_words
    return emotionality

#unused in study
#return average intensity of each emotion in text
#avg = total intensity of emotion / number of emotion words
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

#unused in study
#return % of each emotion in text
#% = total intensity of emotion / total intensity of all emotions
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

#return number of words in text that appear in lexicon
def get_emoword_cnt(data,lex):
    emotion_words = data.apply(lambda x: get_emotion_words(x, lex))
    for i in range(len(emotion_words)):
        emotion_words[i] = len(emotion_words[i])
    return emotion_words

def get_emotion_features (data, lex):
    emotion_features = dict()
    raw_scores = data.apply(lambda x: build_raw_score(x, lex))
    for emotion in raw_scores[0]:
        emotion_features[emotion] = raw_scores.apply(lambda x: x[emotion])
    return emotion_features
    
def main():
    lex = build_lexicon()

if __name__ == "__main__":
    main()