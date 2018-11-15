import keras
from keras.layers import Input, LSTM, Dense, GRU, Embedding, Bidirectional, merge
from keras.models import Model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import html
import numpy as np

HIDDEN_UNITS = 150
TWEET_LENGTH = 170
EMBEDDING_DIM = 50


def read_data(path_template):
    tweets = []
    for i in range(1, 101):
        with open(path_template.format(i), "r") as f:
            inputs = f.readlines()
        tweets.append(inputs)
    print('Found %s authors.' % len(tweets))
    return tweets

def read_embeddings(path):
    embeddings_index = {}
    with open(path, "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def tokenize_tweets(l):
    import sys
    sys.path.append(".")
    import twokenize

    return [twokenize.tokenize(x) for x in l]

def build_word_index(tweets):
    import sys
    sys.path.append(".")
    import twokenize

    tokenized_tweets = [tokenize_tweets(x) for x in tweets]
    flat_vocab = [token for author in tokenized_tweets for tweet in author for token in tweet]
    flat_vocab = list(set(flat_vocab))
    print('Found %s unique tokens.' % len(flat_vocab))

    word_index = {}
    for index, word in enumerate(flat_vocab, start=1):
        word_index[word] = index
    return word_index

def build_embedding_matrix(embeddings_index, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, index in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[index] = embedding_vector
    ratio_not_found = 100. * len(np.where(~embedding_matrix.any(axis=1))[0]) / len(embedding_matrix)
    print("{}% tokens not found in word index.".format(ratio_not_found))
    return embedding_matrix

def pre_process_tweets(tweets):
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
    
def pre_process_pairs(tweet_pairs):
    tokenized_a, tokenized_b = tokenize_tweets(tweet_pairs[0]), tokenize_tweets(tweet_pairs[1])
    pre_a, pre_b = pre_process(tokenized_a), pre_process(tokenized_b)
    pad_a, pad_b = pad_sequences(pre_a, maxlen=TWEET_LENGTH), pad_sequences(pre_b, maxlen=TWEET_LENGTH)
    labels = tweet_pairs[2]
    return pad_a, pad_b, labels

def build_qian_model(vocab_size, embedding_matrix):
    tweet_a = Input(shape=(TWEET_LENGTH,))
    tweet_b = Input(shape=(TWEET_LENGTH,))

    # input = AxB, A = size of input (number of tweets), B = size of vocabulary
    # Weights = NxM, N = size of vocabulary, M = dimensions
    emb = Embedding(input_dim=vocab_size + 1,
                    output_dim=EMBEDDING_DIM,
                    trainable=True,
                    input_length=TWEET_LENGTH,
                    mask_zero=True, 
                    eights=[embedding_matrix]) 
    gru = GRU(HIDDEN_UNITS)

    encoded_a = gru(emb(tweet_a))
    encoded_b = gru(emb(tweet_b))

    cosine = keras.layers.dot([encoded_a, encoded_b], axes=-1, normalize=True)

    predictions = Dense(1, activation='sigmoid')(cosine)

    clf = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

    clf.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return clf

def read_pairs(path):
    # Placeholder
    return [["a"], ["b"], [True]]

# TODO: Add command-line args
def main():
    print("Reading tweet data...")
    tweets = read_data("")
    print("Done!")

    print("Reading embeddings...")
    embeddings = read_embeddings("")
    print("Done!")

    print("Building boilerplate...")
    word_index = build_word_index(tweets)
    embedding_matrix = build_embedding_matrix(embeddings, word_index)
    print("Done!")

    print("Creating model...")
    model = build_qian_model(len(word_index), embedding_matrix)
    print("Done!")
    
    pairs = read_pairs("")
    clf.fit([pairs[0], pairs[1]], pairs[2], epochs=12)
    
    # Re-training test
    y_pred = [1 if x > 0.5 else 0 for x in clf.predict([pairs[0], pairs[1]])]
    y_train = list(map(int, pairs[2]))
    retraining_acc = (np.array(y_pred) == np.array(y_train)).sum() / len(y_pred)
    print (retraining_acc)
    
if __name__ == "__main__":
    main()