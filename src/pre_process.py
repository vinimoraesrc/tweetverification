import copy
import sys
import re
from functools import reduce
import argparse
import operator
import numpy as np
from regexes import preprocess_glove_ruby_port_authors
from keras.preprocessing.sequence import pad_sequences
from contractions import remove_contractions_from_list

tweet_length = 18
embedding_dim = 50


def read_tweets(path):
    path_template = path + "tweet_data_{}.txt"
    tweets = []
    for i in range(1, 101):
        with open(path_template.format(i), "r") as f:
            inputs = f.readlines()
        tweets.append(inputs)
    return tweets


def build_embeddings_index(path):
    embeddings_index = {}
    with open(path, "r") as f:
        for line in f:
            word, *coefs = line.split()
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def remove_rts(tweets):
    tweets_without_RT = []
    for author in tweets:
        tweets_without_RT.append(
            [tweet for tweet in author if not tweet.startswith("RT")])
    return tweets_without_RT


def tokenize_tweets(tweets):
    sys.path.append('.')
    import twokenize
    decoded = [x.replace("\\n", "\n") for x in tweets]
    ttweets = [twokenize.tokenizeRawTweetText(x) for x in decoded]
    uncased = []
    for tokens in ttweets:
        uncased.append([x.lower() for x in tokens])
    return uncased


def split_num(text):
    m = re.search(r"(.*)(<number>)(.*)", text)
    num = m.group(2)
    return [x for x in [m.group(1).strip(), num.strip(), m.group(3).strip()]
            if len(x)]


def treat_num(text):
    return [text.strip()] if "<number>" not in text else \
        [x for x in split_num(text) if len(x)]


def explode_nums(texts):
    exploded = [treat_num(x) for x in texts]
    return reduce(operator.add, exploded)


def explode_nums_from_list(texts_list):
    return [explode_nums(texts) for texts in texts_list if len(texts) > 0]


def split_hashtag(text):
    m = re.search(r"(.*)(<hashtag>)(.*)", text)
    hashtag = m.group(2)
    return [x for x in [m.group(1).strip(), hashtag.strip(), m.group(3).strip()]
            if len(x)]


def treat_hashtag(text):
    return [text.strip()] \
        if "<hashtag>" not in text \
        else [x for x in split_hashtag(text) if len(x)]


def explode_hashtag(texts):
    exploded = [treat_hashtag(x) for x in texts]
    return reduce(operator.add, exploded)


def explode_hashtag_from_list(texts_list):
    return [explode_hashtag(texts) for texts in texts_list if len(texts) > 0]


def split_elong(text):
    m = re.search(r"(.*)(<elong>)(.*)", text)
    elong = m.group(2)
    return [x for x in [m.group(1).strip(), elong.strip(), m.group(3).strip()]
            if len(x)]


def treat_elong(text):
    return [text.strip()] \
        if "<elong>" not in text \
        else [x for x in split_elong(text) if len(x)]


def explode_elong(texts):
    exploded = [treat_elong(x) for x in texts]
    return reduce(operator.add, exploded)


def explode_elong_from_list(texts_list):
    return [explode_elong(texts) for texts in texts_list if len(texts) > 0]


def split_dash(text):
    return [x.strip() for x in text.split("/")]


def treat_dash(text):
    return [text.strip()] \
        if "/" not in text \
        else [x for x in split_dash(text) if len(x)]


def explode_dash(texts):
    exploded = [treat_dash(x) for x in texts]
    return reduce(operator.add, exploded)


def explode_dash_from_list(texts_list):
    return [explode_dash(texts) for texts in texts_list if len(texts) > 0]


def perform_all_filters(tts):
    filt = preprocess_glove_ruby_port_authors(tts)
    filt = explode_nums_from_list(filt)
    filt = explode_hashtag_from_list(filt)
    filt = explode_elong_from_list(filt)
    filt = explode_dash_from_list(filt)
    filt = remove_contractions_from_list(filt)
    return filt


def build_embeddings_matrix(embeddings_index, word_index):
    unmatched_words = []
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, index in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[index] = embedding_vector
        else:
            unmatched_words.append(word)
    ratio_not_found = (
        100. * len(np.where(~embedding_matrix.any(axis=1))[0]) - 1) / len(embedding_matrix)
    print("{}% tokens not found in embedding index.".format(ratio_not_found))
    return embedding_matrix


def pre_process(tweets, word_index):
    indexed_tweets = []
    for tweet in tweets:
        indexed_tweet = []
        for token in tweet:
            if token in word_index:
                indexed_tweet.append(word_index[token])
            else:
                indexed_tweet.append(0)
        indexed_tweets.append(indexed_tweet)
    return indexed_tweets


def pre_process_pairs(tweet_pairs, word_index):
    tokenized_a, tokenized_b = perform_all_filters(
        tokenize_tweets(tweet_pairs[0])), perform_all_filters(
        tokenize_tweets(tweet_pairs[1]))
    pre_a, pre_b = pre_process(
        tokenized_a, word_index), pre_process(
        tokenized_b, word_index)
    pad_a, pad_b = pad_sequences(
        pre_a, maxlen=tweet_length), pad_sequences(
        pre_b, maxlen=tweet_length)
    labels = tweet_pairs[2]
    return pad_a, pad_b, labels


def pre_processing_pipeline(tweets_path, embeddings_path):
    tweets = read_tweets(tweets_path)
    tweets_without_RT = remove_rts(tweets)
    tokenized_tweets = [tokenize_tweets(x) for x in tweets_without_RT]
    tokenized_filtered = [perform_all_filters(x) for x in tokenized_tweets]
    flat_vocab = [
        token for author in tokenized_filtered for tweet in author
        for token in tweet]
    flat_vocab = list(set(flat_vocab))
    word_index = {}
    for index, word in enumerate(flat_vocab, start=1):
        word_index[word] = index
    embeddings_index = build_embeddings_index(embeddings_path)
    embeddings_matrix = build_embeddings_matrix(embeddings_index, word_index)
    return word_index, embeddings_matrix
