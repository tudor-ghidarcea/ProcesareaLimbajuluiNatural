import json
from num2words import num2words
import re
import spacy
"!python -m spacy download ro_core_news_sm"
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
import matplotlib.pyplot as plt
import numpy as np
import nltk

nltk.download('punkt')

# 1:
print("Positive reviews:")

# reading from negative_reviews.json
file = open('positive_reviews.json')
positive_data = json.load(file)

# extracting the content of every review and creating the corpus
file.close()
positive_corpus=[]
for i in positive_data['reviews']:
    positive_corpus.append(i['content'])
    
# displaying the characters different from lowercase letters
positive_chars = set()
for word in positive_corpus:
    for character in word:
        if not character.islower() and not character.isspace() and character not in positive_chars:
            positive_chars.add(character)
print("Characters different to lowercase letters::")
print(positive_chars)

# transforming numbers using num2words
positive_reviews_num = []
for word in positive_corpus:
    positive_reviews_num.append(' '.join([num2words(word,lang='ro') if word.isdigit() else word for word in word.split()]))
print("The content after transforming numbers into words:")
print(positive_reviews_num)

# removing links and other references
reviews_no_links_positive = [re.sub(r'http\S+', '', word) for word in positive_reviews_num]
print("The content after removing links:")
print(reviews_no_links_positive)

# removing punctuation marks
reviews_no_punct_positive = [re.sub(r'[^\w\s]', '', word) for word in reviews_no_links_positive]
print("Content after removing punctuation marks:")
print(reviews_no_punct_positive)

# split the text into tokens
positive_tokenised_review = [word_tokenize(word) for word in reviews_no_punct_positive]
print("All tokens/words:")
print(positive_tokenised_review)

# remove stopwords
nlp = spacy.load('ro_core_news_sm')
positive_stopwords = nlp.Defaults.stop_words
all_words_positive = [word for sent in positive_tokenised_review for word in positive_stopwords]
positives_nostop = [word for word in all_words_positive if word not in positive_stopwords]

print("All words after removing stopwords:")
print(positives_nostop)

# stemming
stemmer = SnowballStemmer(language='romanian')
print("All words after stemming:")
for token in all_words_positive:
    print(token, '=>' , stemmer.stem(token))
print("All words after lematising:")

# lematising
doc = nlp(' '.join(all_words_positive))
for token in doc:
    print(token, '=>', token.lemma_)

# comparing results from stemming and lematising; displaying the top 15 words with different results
stem_diff_lema = []
tokens_proc = set()

for token in doc:
    token_string = token.text
    if token_string in tokens_proc:
        continue
    stem = stemmer.stem(token_string)
    lemma = token.lemma_
    if stem != lemma:
        diff_chars = sum([stem[i]!=lemma[i] for i in range(min(len(stem),len(lemma)))])
        stem_diff_lema.append((token.text, diff_chars))
        tokens_proc.add(token_string)

diff_stem_lemma_sorted = sorted(stem_diff_lema, key=lambda x: -x[1])

print("Top 15 words with different stemming from lematising:")
for token, diff_chars in diff_stem_lemma_sorted[:15]:
    print(token)

#top 20 trigrams
trigram_measures = TrigramAssocMeasures()
finder_collocation = TrigramCollocationFinder.from_words(all_words_positive)
trigrams = finder_collocation.nbest(trigram_measures.pmi, 20)

print("Top 20 trigrams:")
print(trigrams)

# 2:

reviews_stopwords_positive = [word for word in all_words_positive if word in positive_stopwords]
values, frequencies = np.unique(reviews_stopwords_positive, return_counts=True)
stopwords = {value: freq for value, freq in zip(values, frequencies)}

#frequency of every token from e) within the text
plt.bar(values, frequencies, orientation='vertical')
plt.xlim(-1, 10)
plt.show()

reviews_stopwords_positive = [word for word in diff_stem_lemma_sorted if word[0] in positive_stopwords]
values, frequencies = np.unique(reviews_stopwords_positive, return_counts=True)
stopwords = {value: freq for value, freq in zip(values, frequencies)}

#frequency of every token from g)
plt.bar(values, frequencies, orientation='vertical')
plt.xlim(-1, 10)
plt.show()

print("Negative reviews:")

# reading from netative_reviews.json and doing all the same processing as with positive_reviews
g = open('negative_reviews.json')
negative_data = json.load(g)
g.close()  #
negative_corpus = []
for i in negative_data['reviews']:
    negative_corpus.append(i['content'])

lst_char_negative = set()
for word in negative_corpus:
    for character in word:
        if not character.islower() and not character.isspace() and character not in lst_char_negative:
            lst_char_negative.add(character)

print("Characters different to lowercase letters:")
print(lst_char_negative)

negatives_num2words = []

for word in negative_corpus:
    negatives_num2words.append(
        ' '.join([num2words(word, lang='ro') if word.isdigit() else word for word in word.split()]))

print("Content after changing umbers into words:")
print(negatives_num2words)

reviews_no_links_negative = [re.sub(r'http\S+', '', word) for word in
                             negatives_num2words]

print("Content after removing links:")
print(reviews_no_links_negative)

reviews_no_punct_negative = [re.sub(r'[^\w\s]', '', word) for word in
                             reviews_no_links_negative]

print("Content after removing punctuation marks:")
print(reviews_no_punct_negative)

negative_tokenised_review = [word_tokenize(word) for word in
                                  reviews_no_punct_negative]

print("All tokens/words:")
print(negative_tokenised_review)

nlp = spacy.load('ro_core_news_sm')
negative_stopwords = nlp.Defaults.stop_words
all_words_negative = [word for sent in negative_tokenised_review for word in negative_stopwords]
negatives_nostop = [word for word in all_words_negative if word not in negative_stopwords]

print("All words after removing stopwords::")
print(negatives_nostop)

stemmer = SnowballStemmer(language='romanian')

print("All words after stemming:")
for token in all_words_negative:
    print(token, '=>', stemmer.stem(token))

print("All words after lematising:")
doc = nlp(' '.join(all_words_negative))
for token in doc:
    print(token, '=>', token.lemma_)

stem_diff_lema = []
tokens_proc = set()

for token in doc:
    token_string = token.text
    if token_string in tokens_proc:
        continue
    stem = stemmer.stem(token_string)
    lemma = token.lemma_
    if stem != lemma:
        diff_chars = sum([stem[i] != lemma[i] for i in range(min(len(stem), len(lemma)))])
        stem_diff_lema.append((token.text, diff_chars))
        tokens_proc.add(token_string)

diff_stem_lemma_sorted = sorted(stem_diff_lema, key=lambda x: -x[1])
print("Top 15 words with different stemming from lematising:")
for token, diff_chars in diff_stem_lemma_sorted[:15]:
    print(token)

trigram_measures = TrigramAssocMeasures()
finder_collocation = TrigramCollocationFinder.from_words(all_words_negative)
trigrams = finder_collocation.nbest(trigram_measures.pmi, 20)

print("Top 20 trigrams:")
print(trigrams)

# 2:

reviews_stopwords_negative = [word for word in all_words_negative if word in negative_stopwords]
values, frequencies = np.unique(reviews_stopwords_negative, return_counts=True)
stopwords = {value: freq for value, freq in zip(values, frequencies)}

# frequency of every token from e)
plt.bar(values, frequencies, orientation='vertical')
plt.xlim(-1, 10)
plt.show()

reviews_stopwords_negative = [word for word in diff_stem_lemma_sorted if word[0] in negative_stopwords]
values, frequencies = np.unique(reviews_stopwords_negative, return_counts=True)
stopwords = {value: freq for value, freq in zip(values, frequencies)}

# frequency of every token from g)
plt.bar(values, frequencies, orientation='vertical')
plt.xlim(-1, 10)
plt.show()

# The first plot is limited to 20 and displays the appearance of every corpus in alphabetical order.
# The second plot displays the frequency of appearance of the first 10 words
# for which stemming amd lematising led to different results.

# 3:

positive_tokens_per_review = [len(review) for review in positive_tokenised_review]
negative_tokens_per_review = [len(review) for review in negative_tokenised_review]

plt.hist(positive_tokens_per_review, bins=50)
plt.xlabel('Number of tokens')
plt.ylabel('Number of reviews')
plt.show()

plt.hist(negative_tokens_per_review, bins=50)
plt.xlabel('Number of tokens')
plt.ylabel('Number of reviews')
plt.show()

# Negative reviews generally have more words than the positive ones.
