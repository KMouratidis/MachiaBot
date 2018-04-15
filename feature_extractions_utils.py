from sqlite_utils import fetch_all
from collections import Counter, defaultdict
import numpy as np
import spacy

from nltk import word_tokenize, sent_tokenize
from nltk import WordNetLemmatizer
from nltk import wordnet as wn
from nltk.collocations import FreqDist, ngrams
from nltk.corpus import stopwords

from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

machiavelli, montesquieu = fetch_all()

stop_words = stopwords.words("english")

# TODO: Refactor code such that each method can be called on arbitrary text, not just on self.attributes.
class Corpus_:

    def __init__(self,):
        """
        The Corpus_ class is to be initialized without arguments. The default use is intended to follow
        a certain pipeline: add corpuses (lists of sentences stored in a default_dict), use methods on
        some, all, or none of them (e.g. on other text), etc.
        """

        self.corpus = defaultdict(list)
        self.tokenized = defaultdict(list)
        self.lemmatized = defaultdict(list)
        self.create_lemmatization_mappings()

        # a unified collection of all sentences/tokens/lemmas of all documents
        self.corpus_library = list()
        self.token_library = list()
        self.lemma_library = list()

        self.vectorizer = None
        self.feature_names = None
        self.topics = None
        self.topic_algorithm = None

    def add_corpus(self, corpus, corpus_name, also_process=True, raw=False, tokenized=False):
        """

        :param corpus:
        :param corpus_name:
        :param also_process:
        :param raw:
        :return:
        """

        assert isinstance(corpus_name, str), "Corpus name is not a string"
        if tokenized:
            assert isinstance(corpus, list), "Input `corpus` is not a list."
            assert (all(isinstance(sent, list) for sent in corpus)), "Sentences are not lists."
        elif not raw:
            assert isinstance(corpus, list), "Input `corpus` is not a list."
            assert (all(isinstance(sent, str) for sent in corpus)), "Sentences are not strings."
        else:
            assert isinstance(corpus, str), "Input `corpus` is not a string."
            corpus = sent_tokenize(corpus)

        self.corpus[corpus_name] = corpus
        self.corpus_library.extend(corpus)

        if also_process:
            if not tokenized:
                self.tokenize_corpus(corpus_name)
            self.lemmatize_corpus(corpus_name)
            self.create_lemmatization_mappings()

    def tokenize_corpus(self, corpus_name, remove_punctuation=True, lower=True, remove_stopwords=False, ret=False):
        """
        Helper method to tokenize a certain corpus with various optional arguments. Tokenized text
        gets stored in the self.tokenized defaultdict and can be accessed with its key.

        :param corpus_name: A string pointing to a saved corpus.
        :param remove_stopwords: Remove common words. Using nltk's stopwords for English. Optional, default=False.
        :param remove_punctuation: Removes punctuation. Optional, default=True.
        :param lower: Lower the text. Optional, default=True.
        :param ret: Set to True if you need the method to return the tokens, default=False.
        :return: If ret=True, tokenized text, can also be accessed as an attribute.
        """

        if lower and remove_punctuation:
            # simple_preprocess lowers, removes punctuation, and tokenizes, all-in-one and much faster
            tokens = [simple_preprocess(sent) for sent in self.corpus[corpus_name]]
        elif lower:
            tokens = [word_tokenize(sent.lower()) for sent in self.corpus[corpus_name]]
        else:
            tokens = [[word_tokenize(word) for word in sent] for sent in self.corpus[corpus_name]]

        if remove_stopwords:
            tokens = [[word for word in sent if word not in stop_words] for sent in tokens]

        self.tokenized[corpus_name] = tokens
        self.token_library.extend(tokens)

        if ret:
            return tokens

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

    def lemmatize_corpus(self, corpus_name, lemmatizer="synset", ret=False):
        # TODO: add more lemmatizers
        """
        Take a corpus's tokenized form and lemmatize each word based
        on selected lemmatizer. If ret is True, values are returned.

        :param tokens: Should be a list of lists with strings, e.g.: [["d", "w"], ["a"]].
        :param lemmatizer: There are various lemmatizers available, passed as string arguments:
            - 'wordnet': Uses nltk.WordNetLemmatizer
            - 'synset': Uses a custom implementation. Takes a word and returns its most common (or simplest) synset
            - 'spacy': Uses spacy.load('en')
        :param ret: Set to True if you need the method to return the lemmatized tokens, default=False.
        :return: List of lists with lemmatized tokens,  e.g.: [["d", "w"], ["a"]].
        """

        if lemmatizer == "wordnet":
            wnl = WordNetLemmatizer()
            _lemmatized = [[wnl.lemmatize(word) for word in sent] for sent in self.tokenized[corpus_name]]
        elif lemmatizer == "spacy":
            nlp = spacy.load('en', disable=['parser', 'ner'])
            allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
            _lemmatized = [[token.lemma_ for token in nlp(" ".join(sent)) if token.pos_ in allowed_postags]
                           for sent in self.tokenized[corpus_name]]
        else:
            # If'synset'
            print("Using 'synset' lemmatizer.")
            _lemmatized = [[self._synset_lemmatizer(word) for word in sent] for sent in self.tokenized[corpus_name]]

        self.lemmatized[corpus_name] = _lemmatized
        self.lemma_library.extend(_lemmatized)

        if ret:
            return _lemmatized

    def create_lemmatization_mappings(self, ret=False):
        """
        If ret is True this method returns a defaultdict where the keys are the words in the
        lemmatized text and the values are lists of words from the original text that were
        transformed to said new word. If ret is False, then the `mappings` attribute gets updated.
        These mappings are joint for all corpuses.

        :return: If ret=True, returns a defaultdict of word mappings, e.g.: {"new":["old1", "old2"]}.
        """

        word_mappings = defaultdict(list)

        for corpus_name, corpus in self.corpus.items():
            for (old, new) in zip([w for sent in self.tokenized[corpus_name] for w in sent],
                                  [w for sent in self.lemmatized[corpus_name] for w in sent]):

                if old not in word_mappings[new]:
                    word_mappings[new].append(old)

        self.mappings = word_mappings

        if ret:
            return word_mappings

    def model_topics(self, corpus_name=None, text=None, interpret=False,
                     train_also_on_lemmatized=False, train_only_on_lemmatized=False,
                     algorithm="lda_sklearn", no_features=3000, no_topics=50, no_top_words=15):
        """
        Take a list of sentences / documents and return a list of topics, and the created model.

        :param corpus_name:
        :param interpret: Whether to print human-readable results. One could also use self.topics.
        :param text: A list of strings / sentences / documents.
        :param train_also_on_lemmatized:
        :param train_only_on_lemmatized:
        :param algorithm:
        :param no_topics: How many topics do we want to keep track of, int, default=50.
        :param no_features: How many words we want to keep track of, int, default=3000.
        :param no_top_words: How many words to keep track of per topic, int, default=15.
        :param ret :return: A list of lists with keywords representing a topic, and the model instance.
        """

        if text is not None:
            assert isinstance(text, list), "Input `corpus` is not a list."

        # TODO: add other ways for topic modelling
        if algorithm == "lda_sklearn":

            if self.topic_algorithm != algorithm:

                self.topic_algorithm = algorithm

                self.vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                                        max_features=no_features,
                                                        stop_words='english')

                # Fit the model with all documents currently available, store sparse counts vector
                if train_also_on_lemmatized:
                    extended_set = [" ".join(sent) for sent in self.lemma_library]
                    extended_set.extend(self.corpus_library)
                    tf_counts = self.vectorizer.fit_transform(extended_set)
                elif train_only_on_lemmatized:
                    tf_counts = self.vectorizer.fit_transform([" ".join(sent) for sent in self.lemma_library])
                else:
                    tf_counts = self.vectorizer.fit_transform(self.corpus_library)

                # Create LDA model and fit it
                self.topic_model = LatentDirichletAllocation(n_components=no_topics, max_iter=30,
                                                learning_method='batch', evaluate_every=128,
                                                perp_tol=1e-3, learning_offset=50.)

                self.topic_model.fit(tf_counts)

                self.feature_names = self.vectorizer.get_feature_names()

                self.topics = [[self.feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
                               for topic in self.topic_model.components_]

            if corpus_name is not None:
                tf_counts = self.vectorizer.transform(self.corpus[corpus_name])
            elif text is not None:
                tf_counts = self.vectorizer.transform(text)
            else:
                return

            topic_scores = self.topic_model.transform(tf_counts)

        elif algorithm == "lda_gensim":

            if self.topic_algorithm != "lda_gensim":
                self.topic_algorithm = algorithm
                if train_also_on_lemmatized:
                    extended_set = list()
                    extended_set.extend(self.lemma_library)
                    extended_set.extend(self.token_library)

                    self.feature_names = Dictionary(extended_set)
                    bow = [self.feature_names.doc2bow(sent) for sent in extended_set]

                elif train_only_on_lemmatized:
                    self.feature_names = Dictionary(self.lemma_library)
                    bow = [self.feature_names.doc2bow(sent) for sent in self.lemma_library]

                else:
                    self.feature_names = Dictionary(self.token_library)
                    bow = [self.feature_names.doc2bow(sent) for sent in self.token_library]

                self.topic_model = LdaModel(corpus=bow, id2word=self.feature_names, num_topics=no_topics,
                                            chunksize=100, passes=10, alpha="auto", per_word_topics=True)

                self.topics = self.topic_model.print_topics()

            if corpus_name is not None:
                bow = [self.feature_names.doc2bow(sent) for sent in self.lemmatized[corpus_name]]

                topic_scores = self.topic_model[bow]

            # TODO: bugfix this: 'TypeError: doc2bow expects an array of unicode tokens on input, not a single string'
            # elif text is not None:
            #     # Text needs to be converted to tokens and then lemmas
            #     text = [word_tokenize(sent) for sent in text]
            #     text = [[self._synset_lemmatizer(word) for word in sent] for sent in text]
            #     text = [" ".join(word for sent in text for word in sent)]
            #     bow = [self.feature_names.doc2bow(sent) for sent in text]
            #
            #     topic_scores = self.topic_model[bow]

        else:
            print("Wrong algorithm.")
            return

        if topic_scores:
            return topic_scores


    def statistics(self, ret=False):
        # TODO: add more
        token_frequency_dict = FreqDist([word for sent in self.tokenized for word in sent])
        lemma_frequency_dict = FreqDist([word for sent in self.lemmatized for word in sent])

        if ret:
            return token_frequency_dict, lemma_frequency_dict
        else:
            self.token_frequency_dict = token_frequency_dict
            self.lemma_frequency_dict = lemma_frequency_dict
