from nltk.tokenize import sent_tokenize
from numpy import std
from collections import Counter

common_elems = [',', ';', '"', '!', '-', 'and', 'but', 'however',
                'if', 'that', 'more', 'must', 'might', 'this', 'very']


def initial_read(text):
    """Reads the text file and saves it as a string"""

    words = text.split()
    sentences = sent_tokenize(text)

    data = list()
    data.append(mean_word_length(words))
    data.append(mean_sentence_length(sentences))
    data.append(sd_of_sentence_length(sentences))

    for word in common_elems:
        data.append(count_words(text, word, len(words)))

    return data


def mean_word_length(words):
    """Calculates mean length of words in a text"""
    total_length = 0
    for word in words:
        total_length += len(word)
    return total_length/len(words)


def mean_sentence_length(sentences):
    """Calculates the mean length of each sentence"""
    total_length = 0
    for sentence in sentences:
        total_length += len(sentence)

    mean = total_length/len(sentences)
    return mean


def sd_of_sentence_length(sentences):
    """Returns the standard deviation in sentence length"""
    sentence_lengths = []
    for sentence in sentences:
        sentence_length = len(sentence)
        sentence_lengths.append(sentence_length)

    sd = std(sentence_lengths)
    return sd


def count_words(text, word, wordcount):
    """Returns the count of a given word/character per 1000 words"""
    total = text.count(word)
    thousands = wordcount / 1000.0
    total /= thousands
    return total


def type_token_ratio(words):
    """Calculates the type token ratio of a text"""
    c = Counter(words)
    # TTR is the number of unique words in a text divided by the total word count
    ttr = float(len(c)) / len(words)
    return ttr

# Other factors are mean paragraph length and chapter length
# Might not be relevant so have omitted for now


