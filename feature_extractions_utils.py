from sqlite_utils import fetch_all
from collections import Counter, defaultdict
import numpy as np

from nltk import word_tokenize, sent_tokenize
from nltk import WordNetLemmatizer
from nltk import wordnet as wn
from nltk.collocations import FreqDist, ngrams
from nltk.corpus import stopwords

from gensim.utils import simple_preprocess

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

machiavelli, montesquieu = fetch_all()

stop_words = stopwords.words("english")


class Corpus_:

    def __init__(self, corpus, raw=False):
        """
        The Corpus_ class is to be initialized with a list of strings representing sentences.
        If input text is a single string, set raw=True.
        """

        if not raw:
            assert isinstance(corpus, list), "Input `corpus` is not a list."
            assert (all(isinstance(sent, str) for sent in corpus)), "Sentences are not strings."
        else:
            assert isinstance(corpus, str), "Input `corpus` is not a string."
            corpus = sent_tokenize(corpus)

        self.corpus = corpus
        self.tokenized = self.tokenize(corpus, ret=True)
        self.lemmatized = self.lemmatize(ret=True)
        self.mappings = self.lemmatization_mappings(ret=True)
        self.topics, self.topic_model = self.get_topics(ret=True)
        self.token_frequency_dict, self.lemma_frequency_dict = self.statistics(ret=True)

    def tokenize(self, remove_stopwords=False, remove_punctuation=True, lower=True, ret=False):
        """
        Helper method to tokenize text with various optional arguments.

        :param remove_stopwords: Remove common words. Using nltk's stopwords for English. Optional, default=False.
        :param remove_punctuation: Removes punctuation. Optional, default=True.
        :param lower: Lower the text. Optional, default=True.
        :param ret: Set to True if you need the method to return the tokens, default=False.
        :return: If ret=True, tokenized text, can also be accessed as an attribute.
        """

        assert (all(isinstance(sent, str) for sent in text)), "List element is not a string."

        if lower and remove_punctuation:
            # simple_preprocess lowers, removes punctuation, and tokenizes, all-in-one and much faster
            tokens = [simple_preprocess(sent) for sent in text]
        elif lower:
            tokens = [word_tokenize(sent.lower()) for sent in text]
        else:
            tokens = [[word_tokenize(word) for word in sent] for sent in text]

        if remove_stopwords:
            tokens = [[word for word in sent if word not in stop_words] for sent in tokens]

        if ret:
            return tokens
        else:
            self.tokenized = tokens

    def _synset_lemmatizer(self, word):
        """
        Simplify words by taking their simpler or most common synsets. Words up to 3 letters do not get modified.

        Examples:
            in: "hello", out: "hello"
            in: "distanced", out: "distance"
            in: "spaces", out: "space"
            in: "told", out: "tell"
        It's not perfect:
            in: "comprehend", out: "grok"
        """

        # don't modify small words
        if len(word) <= 3:
            return word

        try:
            # get synsets
            synsets_list = wn.wordnet.synsets(word)

            # clear synsets: get names as strings
            synsets_list = [w.name().split(".")[0] for w in synsets_list]

            word_counter = Counter(synsets_list)

            # if there are many words
            if len(word_counter) > 1:
                word_freq1, word_freq2 = word_counter.most_common(2)  # each is a tuple: ("word", counts)

                # if they have the same frequencies: pick the shorter word, else pick the first
                if word_freq1[1] == word_freq2[1]:
                    if len(word_freq1[0]) <= len(word_freq2[0]):
                        return word_freq1[0]
                    else:
                        return word_freq2[0]
                else:
                    return word_freq1[0]

            # if there is only one word
            else:
                return word_counter.most_common()[0][0]

        # if there are no synsets, return the word as it is
        except IndexError:
            return word

    def lemmatize(self, lemmatizer="synset", ret=False):
        # TODO: add more lemmatizers
        """
        Take a corpus's tokenized form and lemmatize each word based
        on selected lemmatizer. If ret is True, values are returned.

        :param tokens: Should be a list of lists with strings, e.g.: [["d", "w"], ["a"]].
        :param lemmatizer: There are various lemmatizers available, passed as string arguments:
            - 'wordnet': Uses nltk.WordNetLemmatizer
            - 'synset': Uses a custom implementation. Takes a word and returns its most common (or simplest) synset
        :param ret: Set to True if you need the method to return the lemmatized tokens, default=False.
        :return: List of lists with lemmatized tokens,  e.g.: [["d", "w"], ["a"]].
        """

        if lemmatizer == "wordnet":
            wnl = WordNetLemmatizer()
            _lemmatized = [[wnl.lemmatize(word) for word in sent] for sent in self.tokenized]
        elif lemmatizer == "synset":
            _lemmatized = [[self._synset_lemmatizer(word) for word in sent] for sent in self.tokenized]
        else:
            # return 'synset'
            print("Lemmatizer not recognized, using 'synset'.")
            _lemmatized = [[self._synset_lemmatizer(word) for word in sent] for sent in self.tokenized]

        if ret:
            return _lemmatized
        else:
            self.lemmatized = _lemmatized


    def lemmatization_mappings(self, lemmatized=self.lemmatized, ret=False):
        """
        If ret is True this method returns a defaultdict where the keys are the words in the
        lemmatized text and the values are lists of words from the original text that were
        transformed to said new word. If ret is False, then the `mappings` attribute gets updated.

        :param lemmatized: List of lists with string tokens, e.g.: [["d", "w"], ["a"]], default: self.lemmatized.
        :return: A defaultdict of word mappings, e.g.: {"new":["old1", "old2"]}.
        """

        word_mappings = defaultdict(list)

        for (old, new) in zip([w for sent in self.tokenized for w in sent],
                              [w for sent in lemmatized for w in sent]):

            if old not in word_mappings[new]:
                word_mappings[new].append(old)

        if ret:
            return word_mappings
        else:
            self.mappings = word_mappings

    def get_topics(self, algorithm="lda", no_features=1500, no_topics=30, no_top_words=10, ret=False):
        """
        Take a list of sentences / documents and return a list of topics, and the created model.

        :param text: A list of strings / sentences / documents.
        :param no_topics: How many topics do we want to keep track of, int, default=30.
        :param no_features: How many words we want to keep track of, int, default=1000.
        :param no_top_words: How many words to keep track of per topic, int, default=10.
        :return: A list of lists with keywords representing a topic, and the model instance.
        """

        # TODO: add other ways for topic modelling
        if algorithm == "lda":

            tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                            max_features=no_features,
                                            stop_words='english')

            # Transform text into sparse counts
            tf = tf_vectorizer.fit_transform(self.corpus)
            # Keep track of original words
            tf_feature_names = tf_vectorizer.get_feature_names()

            # Create LDA model and fit it
            model = LatentDirichletAllocation(n_components=no_topics, max_iter=30,
                                            learning_method='batch', evaluate_every=128,
                                            perp_tol=1e-3, learning_offset=50.).fit(tf)

            topics = [[tf_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]] for topic in model.components_]

        if ret:
            return topics, model
        else:
            self.topics, self.topic_model = topics, model


    def statistics(self, ret=False):
        # TODO: add more
        token_frequency_dict = FreqDist([word for sent in self.tokenized for word in sent])
        lemma_frequency_dict = FreqDist([word for sent in self.lemmatized for word in sent])

        if ret:
            return token_frequency_dict, lemma_frequency_dict
        else:
            self.token_frequency_dict = token_frequency_dict
            self.lemma_frequency_dict = lemma_frequency_dict
