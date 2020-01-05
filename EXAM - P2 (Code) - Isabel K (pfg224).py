#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Isabel Kirkegaard (pfg224) - Portfolio 2
"""

#1. The collection
#I have chosen to use the Guardian OpenAPI. Furthermore I wanted results
# from the first of january this year to late october.

#First I follow the tutorial of the Guardian API key and exploring page,
# which I need to import to python

#Import the following libraries:
import json
import requests
import os
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta

# This creates two subdirectories called "theguardian" and "collection"
ARTICLES_DIR = join('theguardian', 'collection')
makedirs(ARTICLES_DIR, exist_ok=True)

# Sample URL

#https://content.guardianapis.com/search?q='youtube'&api-key=2656604b-a2f7-4c73-982a-25c8444df944

MY_API_KEY = '2656604b-a2f7-4c73-982a-25c8444df944'

API_ENDPOINT = 'http://content.guardianapis.com/search'
my_params = {
    'from-date': "", # leave empty, change start_date / end_date variables instead
    'to-date': "",
    'order-by': "newest",
    'show-fields': 'all',
    'page-size': 200,
    'api-key': MY_API_KEY
}

# Update these dates to suit your own needs.
start_date = date(2019, 1, 1)
end_date = date(2019,10, 29)

dayrange = range((end_date - start_date).days + 1)
for daycount in dayrange:
    dt = start_date + timedelta(days=daycount)
    datestr = dt.strftime('%Y-%m-%d')
    fname = join(ARTICLES_DIR, datestr + '.json')
    if not exists(fname):
        # then let's download it
        print("Downloading", datestr)
        all_results = []
        my_params['from-date'] = datestr
        my_params['to-date'] = datestr
        current_page = 1
        total_pages = 1
        while current_page <= total_pages:
            print("...page", current_page)
            my_params['page'] = current_page
            resp = requests.get(API_ENDPOINT, my_params)
            data = resp.json()
            all_results.extend(data['response']['results'])
            # if there is more than one page
            current_page += 1
            total_pages = data['response']['pages']

        with open(fname, 'w') as f:
            print("Writing to", fname)

            # re-serialize it for pretty indentation
            f.write(json.dumps(all_results, indent=2))
import json
import os

# Update to the directory that contains your json files
# Note the trailing /
directory_name = "theguardian/collection/"

ids = list()
texts = list()
sections = list()
for filename in os.listdir(directory_name):
    if filename.endswith(".json"):
        with open(directory_name + filename) as json_file:
            data = json.load(json_file)
            for article in data:
                id = article['id']
                fields = article['fields']
                text = fields['bodyText'] if fields['bodyText'] else ""
                ids.append(id)
                texts.append(text)
                section = article['sectionId']
                sections.append(section)

print("Number of ids: %d" % len(ids))
print("Number of texts: %d" % len(texts))




#2. Pre-process and describe your collection

#Makes the sections into a list to use it for the rest of the tasks. 
#A set also stops duplicates, so it will give a more clear result
sectionslist = set(sections)

#same with ids
idslist = set(ids)

textlist = set(texts)


#import nltk to use the stopwords feature. 
import nltk
nltk.download('stopwords')

#To transform the data into a document-term matrix use CountVectorizer 
#from sklearn and stopwords from nltk

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords as sw
model_vect = CountVectorizer(stop_words=sw.words('english'), token_pattern=r'[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vect = model_vect.fit_transform(texts)
data_vect
#Stop words are regular used words like 'a' and 'in'. These, and 
#other similar have been removed 
#22924x98 sparse matrix


#Finding info about data like uniqueness of a section, or the total
# number of characters
#Number of articles in the most unique sections of the Guardian API 
import numpy as np
unique, counts = np.unique(texts, return_counts=True)
dict(zip(unique, counts))


#The total number of characters in the data of the sections
total = 0
for text in texts:
    total = total + len(text)
total
#total = 187195

#Finding the texts length
all_lengths = list()
for text in texts:
    all_lengths.append(len(text))
print("Total sum: %i" % sum(all_lengths))
#Total sum is 118100885


#How many words are in the data? Using word count
word_count = 0
for text in texts:
    words = text.split()
    word_count = word_count + len(words)
word_count
# total word count = 19999514

#Making all_words into a list and finding the unique word count
all_words = list()
for text in texts:
    words = text.split()
    all_words.extend(words)
unique_words = set(all_words)
unique_word_count = len(unique_words)
print("Unique word count: %i" % unique_word_count)

#Using random to show a sample of unique words
import random
random.sample(unique_words, 30)
#Finds completetly random words

from nltk.probability import FreqDist
fdist = FreqDist(sections)
fdist.most_common(40)


from nltk import word_tokenize

tokens = list()
for text in texts:
  tokens_in_text = word_tokenize(text)
  for token in tokens_in_text:
    if token.isalpha():
      tokens.append(token.lower())

print("Previous word count: %i" % word_count)
print("Word count: %i" % len(tokens))
unique_tokens = set(tokens)
print("Previous unique word count: %i" % unique_word_count)
print("Unique word count: %i" % len(unique_tokens))

counts = data_vect.sum(axis=0).A1
top_idxs = (-counts).argsort()[:10]
top_idxs



#------------

#3. Select articles using a query
#In this case I want to use 'youtube' as a query

term =['Scotland']
term 


term_idxs = [model_vect.vocabulary_.get(term) for term in term]
term_counts = [counts[idx] for idx in term_idxs]
term_counts

inverted_vocabulary = dict([(idx, word) for word, idx in model_vect.vocabulary_.items()])
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
print("Top words: %s" % top_words)


import random
some_row_idxs = random.sample(range(0,len(texts)), 10)
print("Selection: (%s x %s)" % (some_row_idxs, top_idxs))
sub_matrix = data_vect[some_row_idxs, :][:, top_idxs].todense()
sub_matrix


from sklearn.feature_extraction.text import TfidfTransformer
model_tfidf = TfidfTransformer()
data_tfidf = model_tfidf.fit_transform(data_vect)

#My query will be my term 'Scotland' as I mentioned earlier
query = " ".join(term)
query

query_vect_counts = model_vect.transform([query])
query_vect = model_tfidf.transform(query_vect_counts)
query_vect

freqs = data_tfidf.mean(axis=0).A1
top_idxs = (-freqs).argsort()[:10].tolist()
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
(top_idxs, top_words)


import pandas as pd 
sub_matrix = data_tfidf[some_row_idxs, :][:,top_idxs].todense()
df = pd.DataFrame(columns=top_words, index=some_row_idxs, data=sub_matrix)
df


idfs = model_tfidf.idf_
term_idfs = [idfs[idx] for idx in term_idxs]
term_idfs


df = pd.DataFrame(columns=['count', 'idf'], index=term, data=zip(term_counts,term_idfs))
df

from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(query_vect, data_tfidf)
sims

sims_sorted_idx = (-sims).argsort()
sims_sorted_idx


import pandas as pd

print("Shape of 2-D array sims: (%i, %i)" % (len(sims), len(sims[0,:])) )
df = pd.DataFrame(data=zip(sims_sorted_idx[0,:], sims[0,sims_sorted_idx[0,:]]), columns=["index", "cosine_similarity"])
df[0:10]

df = pd.DataFrame(data=zip(query, sims[0,:]), columns=["class", "cosine_similarity"])
df = df.groupby("class").mean()
df = df.assign(name=query)
df = df.sort_values("cosine_similarity", ascending=False)
df[0:5]


from sklearn.decomposition import LatentDirichletAllocation
model_lda = LatentDirichletAllocation(n_components=4, random_state=0)
data_lda = model_lda.fit_transform(data_vect)
# Describe the shape of the resulting matrix.
import numpy as np
np.shape(data_lda)


for i, term_weights in enumerate(model_lda.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = ["%s (%.3f)" % (model_vect.get_feature_names()[idx], term_weights[idx]) for idx in top_idxs]
    print("Topic %d: %s" % (i, ", ".join(top_words)))



#The indexes of the most used words
counts = data_vect.sum(axis=0).A1
top_idxs = (-counts).argsort()[:50]
top_idxs

import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()

ranks = range(1, len(fdist) + 1)
# range of 1-2-3-4-5-...-N 
#freqs = list()
#for token in fdist.keys():
#  freqs.append(fdist[token])
# unsorted list of frequencies per word
#ranks = range(1, fdist.B() + 1)
freqs = list(fdist.values())
# sorted (=ranked!) list of frequencies 
freqs.sort(reverse = True)
plt.plot(ranks, freqs, '-')
plt.xscale('log')
plt.yscale('log')
plt.show()

import random
random.sample(unique_tokens, 20)


all_words = list()
for text in sections:
  words = text.split()
  all_words.extend(words)
print("Word count: %i" % len(all_words))
unique_words = set(all_words)
unique_word_count = len(unique_words)
print("Unique word count: %i" % unique_word_count)


inverted_vocabulary = dict([(idx, word) for word, idx in model_vect.vocabulary_.items()])
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
print("Top words in sections: %s" % top_words)

import random
some_row_idxs = random.sample(range(0,len(ids)), 10)
print("Selection: (%s x %s)" % (some_row_idxs, top_idxs))
sub_matrix = data_vect[some_row_idxs, :][:, top_idxs].todense()
sub_matrix


#-----------------
#4. Model and vizualize the topics in yout subset


#Topic modeling
from sklearn.decomposition import LatentDirichletAllocation
model_lda = LatentDirichletAllocation(n_components=6, random_state=0)
data_lda = model_lda.fit_transform(data_vect)
np.shape(data_lda)

for i, term_weights in enumerate(model_lda.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = ["%s (%.3f)" % (model_vect.get_feature_names()[idx], term_weights[idx]) for idx in top_idxs]
    print("Topic %d: %s" % (i, ", ".join(top_words)))


import pandas as pd
topic_names = ["Topic" + str(i) for i in range(model_lda.n_components)]
doc_names = ["Doc" + str(i) for i in range(len(ids))]
df = pd.DataFrame(data=np.round(data_lda, 2), columns=topic_names, index=doc_names).head(10)
# extra styling
df.style.applymap(lambda val: "background: yellow" if val>.3 else '', )
print(df)


#Using wordcloud to vizualize the most commonly used terms of my query
from wordcloud import WordCloud
import matplotlib.pyplot as plt

for i, term_weights in enumerate(model_lda.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = [model_vect.get_feature_names()[idx] for idx in top_idxs]
    word_freqs = dict(zip(top_words, term_weights[top_idxs]))
    wc = WordCloud(background_color="white",width=300,height=300, max_words=10).generate_from_frequencies(word_freqs)
    plt.subplot(2, 2, i+1)
    plt.imshow(wc)