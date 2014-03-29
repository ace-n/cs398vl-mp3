# FOR GRAPH GENERATION: http://networkx.lanl.gov/index.html
import json
import nltk
from nltk.corpus import brown, PlaintextCorpusReader
from nltk.corpus import wordnet as wn
import cPickle
import copy
import math
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import RegexpStemmer

# --------------- Constants ---------------
# File to work with
file_path = 'Punctuated/ofk_ch4_{!s}.txt'.format("2")

# Laplacian smoothing value (the greater this is, the greater its effect and the less the actual data matters)
smooth_start = 10 # Initial number of occurrences each word should start out with

# Positive/negative (a random sample of the English language)
# TODO find a way to limit the impact of BS words e.g. good/bad
words_positive = ["trust", "trusted", "honest", "friend", "companion", "faithful",
                  "loyal", "dedicated", "love", "honorable", "benevolent", "generous",
                  "kind", "graciously", "glad", "happy"]

words_negative = ["disloyal", "rude", "cruel", "evil", "despicable", "calamity", "scramble",
                  "incompetent", "coward", "farce", "moron", "ignorance", "regret", "vulgar"
                  "unfaithful", "mad", "angry", "irritate", "malevolent", "wicked", "melancholy",
                  "sad"]

# Decide which type of stemmer to use
def stemmer():
   #return RegexpStemmer('ing$|s$|e$', min=4)
   return LancasterStemmer()

# ----------- Actually do stuff -----------
# Train the classifier on the text (step 1)
def train():

   wordlists = PlaintextCorpusReader('', file_path)

   st = stemmer()
   
   # Get blocks of text using NLTK
   words = wordlists.words(file_path)
   sents = wordlists.sents(file_path)
   paras = wordlists.paras(file_path)

   # LOGIC
   #       If a sentence contains a known [posi/nega]tive word, count the instances of words in that sentence as 
   #       [posi/nega]tive

   # Count words
   word_features = []

   # Go through paragraphs
   for p in paras:

      # Classify S
      score_positive_negative = 0
      for s in p:
         for word in s:

            word = st.stem(word)

            if word in words_positive:
               score_positive_negative += 1
            elif word in words_negative:
               score_positive_negative -= 1
   
      # Record class of paragraph for any words present
      for s in p:
         for word in s:

            word = st.stem(word)

            if score_positive_negative > 0:
               word_features.append( ({"word": word}, "+") )
            elif score_positive_negative < 0:
               word_features.append( ({"word": word}, "-") )
            else:
               word_features.append( ({"word": word}, " ") )

   # Create and return classifier
   classifier = nltk.NaiveBayesClassifier.train(word_features)
   return classifier

# Testing
def main():

   st = stemmer()

   # Get data
   wordlists = PlaintextCorpusReader('', file_path)
   words = wordlists.words(file_path)
   sents = wordlists.sents(file_path)
   paras = wordlists.paras(file_path)

   # Train
   classifier = train()

   # Get class probabilities (for MAP estimation)
   counts = {"P":0, "-":0, "N":0}
   for i in range(0,len(paras)):
      for s in paras[i]:

         score_pos = 0
         score_neg = 0

         # Classify paragraph
         for word in s:

            word = st.stem(word)

            feature = {"word":word}
            classified = classifier.classify(feature)

            if classified == "+":
               score_pos += 1
            elif classified == "-":
               score_neg += 1

         # Record result
         if score_pos > score_neg:
            counts["P"] += 1
         elif score_pos < score_neg:
            counts["N"] += 1
         else:
            counts["-"] += 1

   # Done!
   print counts

# Do something
main()