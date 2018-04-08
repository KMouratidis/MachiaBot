from sqlite_utils import fetch_all
from collections import Counter, defaultdict
import numpy as np

from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk import wordnet as wn
from nltk.collocations import FreqDist, ngrams
from nltk.corpus import stopwords

from gensim.utils import simple_preprocess

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

machiavelli, montesquieu = fetch_all()

stop_words = stopwords.words("english")


def _basic_assertions(corpus, advanced_assertions=False):
    """
    Every subsequent function will use this to check if the input is in the correct format.
    The argument advanced_assertions is meant for functions that expect a list of lists with string tokens.
    """

    assert isinstance(corpus, list), "Input `corpus` is not a list."

    # TODO: Should other data types be included for tokens, somehow?
    if advanced_assertions:
        assert (all(isinstance(sent, list) for sent in corpus)), "Sentences are not lists."
        assert (all(isinstance(token, str) for sent in corpus for token in sent)), "Tokens are not strings."


def tokenize(text, remove_stopwords=False, remove_punctuation=True, lower=True):
    """
    Helper function to tokenize text with various optional arguments.

    :param text: Should be a list of strings (sentences, documents, etc),
           e.g.: [["Going a trip around town."], ["It is a wonderful experience."]].
    :param remove_stopwords: Remove common words. Using nltk's stopwords for English. Optional, default=False.
    :param remove_punctuation: Removes punctuation. Optional, default=True.
    :param lower: Lower the text. Optional, default=True.
    :return: Tokenized text, a list of lists with tokens, e.g.: [["d", "w"], ["a"]].
    """

    # Check that input is correct
    _basic_assertions(text)
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

    return tokens


def _synset_lemmatizer(word):
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


def lemmatize(corpus, lemmatizer="synset"):
    # TODO: add more lemmatizers
    """
    Take a corpus and return it in the same format with each word lemmatized.

    :param tokens: Should be a list of lists with strings, e.g.: [["d", "w"], ["a"]].
    :param lemmatizer: There are various lemmatizers available, passed as string arguments:
        - 'wordnet': Uses nltk.WordNetLemmatizer
        - 'synset': Uses a custom implementation. Takes a word and returns its most common (or else simplest) synset.
    :return: List of lists with lemmatized tokens,  e.g.: [["d", "w"], ["a"]].
    """

    # Check that input is correct
    _basic_assertions(corpus, advanced_assertions=True)

    if lemmatizer == "wordnet":
        wnl = WordNetLemmatizer()
        lemmatized_tokens = [[wnl.lemmatize(word) for word in sent] for sent in corpus]
    elif lemmatizer == "synset":
        lemmatized_tokens = [[_synset_lemmatizer(word) for word in sent] for sent in corpus]
    else:
        # return 'synset'
        print("Lemmatizer not recognized, using 'synset'.")
        lemmatized_tokens = [[_synset_lemmatizer(word) for word in sent] for sent in corpus]

    return lemmatized_tokens


def synset_lemmatization_mappings(original, lemmatized):
    """
    Helper function that takes a text transformed with synset lemmatization and returns a defaultdict where
    the keys are the words in the lemmatized text and the values are lists of words from the original text
    that were transformed to said new word.

    :param original: List of lists with string tokens, e.g.: [["d", "w"], ["a"]].
    :param lemmatized: List of lists with string tokens, e.g.: [["d", "w"], ["a"]].
    :return: A defaultdict of word mappings, e.g.: {"new":["old1", "old2"]}.
    """
    for corpus in [original, lemmatized]:
        _basic_assertions(corpus, advanced_assertions=True)

    word_mappings = defaultdict(list)

    for (old, new) in zip([w for sent in original for w in sent],
                          [w for sent in lemmatized for w in sent]):

        if old not in word_mappings[new]:
            word_mappings[new].append(old)

    return word_mappings

# TODO: Maybe convert to class (the whole file?) since it will have a better interface and less programming headache
def get_topics(text, algorithm="lda", no_features=1500, no_topics=30, no_top_words=10):
    """
    Take a list of sentences / documents and return a list of topics, and the created model.

    :param text: A list of strings / sentences / documents.
    :param no_topics: How many topics do we want to keep track of, int, default=30.
    :param no_features: How many words we want to keep track of, int, default=1000.
    :param no_top_words: How many words to keep track of per topic, int, default=10.
    :return: A list of lists with keywords representing a topic, and the model instance.
    """

    # TODO: add other ways for topic modelling
    _basic_assertions(text)
    assert (all(isinstance(sent, str) for sent in text)), "List element is not a string."

    if algorithm == "lda":

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=no_features,
                                        stop_words='english')

        # Transform text into sparse counts
        tf = tf_vectorizer.fit_transform(text)
        # Keep track of original words
        tf_feature_names = tf_vectorizer.get_feature_names()

        # Create LDA model and fit it
        lda = LatentDirichletAllocation(n_components=no_topics, max_iter=30,
                                        learning_method='batch', evaluate_every=128,
                                        perp_tol=1e-3, learning_offset=50.).fit(tf)

        topics = [[tf_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]] for topic in lda.components_]

        return topics, lda



def statistics(corpus):
    # TODO: add more
    frequency_dict = FreqDist([word for sent in corpus for word in sent])

    return frequency_dict
