# Development Set-up
## Prerequisites
* Python (tested on version 3.11.3)
* Packages
  * numpy
  * pandas
  * sklearn
  * nltk
  * matplotlib
## File Overview
* emotions.py - contains functions for emotion feature extraction using the NRC Emotion Intensity Lexicon
* tokenizer.py - contains generic preprocessing functions
* ngrams_svc1.py - standard implementation of linearsvc model with n-grams and emotion features, used for basic data visualization and feature importance
* ngrams_svc_kfold.py - stratified kfold implementation of linearsvc model with n-grams and emotion features, used for model stats and t-test
## Instructions
1. Install the required packages
2. Run 'ngrams_svc1.py' or 'ngrams_svc_kfold.py' using the following command:
```
py file_of_choice
```
