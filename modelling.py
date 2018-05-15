import feature_extractions_utils
from sqlite_utils import fetch_all

from collections import Counter, defaultdict
import numpy as np
import spacy

from nltk import word_tokenize, sent_tokenize
from nltk import WordNetLemmatizer
from nltk import wordnet as wn
from nltk.collocations import FreqDist, ngrams
from nltk.corpus import stopwords
from nltk.corpus import (brown, conll2000, conll2002, conll2007, gutenberg, genesis,
                         europarl_raw.english, reuters, treebank, inaugural, abc, shakespeare)

from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

machiavelli, montesquieu = fetch_all()

stop_words = stopwords.words("english")

# More corpuses => more diversity
corps = (brown, conll2000, conll2002, conll2007, gutenberg, genesis,
         europarl_raw.english, reuters, treebank, inaugural, abc, shakespeare)

corp_names = "brown, conll2000, conll2002, conll2007, gutenberg, genesis, europarl_raw.english, " \
             "reuters, treebank, inaugural, abc, shakespeare".split(", ")

corpus = feature_extractions_utils.Corpus_()

# TODO: preprocess these corpuses before adding them
for corp, name in zip(corps, corp_names):
    corpus.add_corpus(list(corp.sents()), name, tokenized=True)

corpus.add_corpus(machiavelli, "machiavelli")
corpus.add_corpus(montesquieu, "montesquieu")

# So far we have:
# print(len(corpus.corpus_library))
# 336.150 sentences
# print(sum([len(sent) for sent in corpus.corpus_library]))
# 8.807.803 words
# Unique words: 229.401 vs 182.564 (-46.837 words, or -20%)
# n_unique_initial = set([word for sent in corpus.corpus_library for word in sent])
# n_unique_final = set([word for sent in corpus.lemma_library for word in sent])


# trains on ALL corpuses' lemmatized form.
corpus.model_topics(train_only_on_lemmatized=True, algorithm="lda_sklearn")