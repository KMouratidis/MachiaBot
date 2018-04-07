from sqlite_utils import fetch_all
from string import punctuation
from collections import Counter, defaultdict
import numpy as np

from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk import wordnet as wn
from nltk.corpus import stopwords

from gensim.utils import simple_preprocess

machiavelli, montesquieu = fetch_all()

stop_words = stopwords.words("english")


def tokenize(text, remove_stopwords=False, remove_punctuation=True, lower=True):
    """
    Helper function to tokenize text with various optional arguments.

    :param text: Should be a list of strings (sentences, documents, etc)
    :param remove_stopwords: Remove common words. Using nltk's stopwords for English. Optional, default=False.
    :param remove_punctuation: Removes punctuation. Optional, default=True
    :param lower: Lower the text. Optional, default=True
    :return: Tokenized text, a list of lists with tokens.
    """

    # Check that input is correct
    assert isinstance(text, list), "Input `text` is not a list."
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


def lemmatize(tokens, lemmatizer="synset"):
    # TODO: add more lemmatizers
    """
    Take a list of lists with tokens and return it in the same format with each word lemmatized.
    There are various lemmatizers available, passed as string arguments:
    - 'wordnet': Uses nltk.WordNetLemmatizer
    - 'synset': Uses a custom implementation. Takes a word and returns its most common (or else simplest) synset.

    :param tokens:
    :param lemmatizer:
    :return:
    """

    # Check that input is correct
    assert isinstance(tokens, list), "Input `text` is not a list."
    assert (all(isinstance(sent, list) for sent in tokens)), "Sentences are not lists."
    assert (all(isinstance(token, str) for sent in tokens for token in sent)), "Tokens are not strings."

    if lemmatizer == "wordnet":
        wnl = WordNetLemmatizer()
        lemmatized_tokens = [[wnl.lemmatize(word) for word in sent] for sent in tokens]
    elif lemmatizer == "synset":
        lemmatized_tokens = [[_synset_lemmatizer(word) for word in sent] for sent in tokens]
    else:
        # return 'synset'
        print("Lemmatizer not recognized, using 'synset'.")
        lemmatized_tokens = [[_synset_lemmatizer(word) for word in sent] for sent in tokens]

    return lemmatized_tokens

def synset_lemmatization_mappings(original, lemmatized):
    """
    Helper function that takes a text transformed with synset lemmatization and returns a defaultdict where
    the keys are the words in the lemmatized text and the values are lists of words from the original text
    that were transformed to said new word.

    :param original: List of lists with string tokens.
    :param lemmatized: List of lists with string tokens.
    :return:
    """
    for tokens in [original, lemmatized]:
        assert isinstance(tokens, list), "Input `text` is not a list."
        assert (all(isinstance(sent, list) for sent in tokens)), "Sentences are not lists."
        assert (all(isinstance(token, str) for sent in tokens for token in sent)), "Tokens are not strings."

    word_mappings = defaultdict(list)

    for (old, new) in zip([w for sent in original for w in sent],
                          [w for sent in lemmatized for w in sent]):

        if old not in word_mappings[new]:
            word_mappings[new].append(old)