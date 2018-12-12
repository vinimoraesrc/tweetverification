import keras
from keras.layers import Input, LSTM, Dense, GRU, Embedding, Bidirectional, merge, average, Lambda
from keras.models import Model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

tweet_length = 18
embedding_dim = 50
hidden_units = 150


def build_model(word_index, embedding_matrix):
    tweet_a = Input(shape=(tweet_length,))
    tweet_b = Input(shape=(tweet_length,))

    emb = Embedding(
        input_dim=len(word_index) + 1, output_dim=embedding_dim, trainable=True,
        input_length=tweet_length, mask_zero=True, weights=[embedding_matrix])
    gru = Bidirectional(GRU(hidden_units))

    encoded_a = gru(emb(tweet_a))
    encoded_b = gru(emb(tweet_b))

    def L2_distance(x): return K.abs(x[0]-x[1]) ** 2

    def merge_all(x): return K.concatenate(
        [x[0], x[1], L2_distance(x), x[0] * x[1]], axis=1)

    merge_layer = keras.layers.Lambda(merge_all)
    merged = merge_layer([encoded_a, encoded_b])

    dense_1 = Dense(hidden_units, activation='relu')(merged)
    dropout = keras.layers.Dropout(0.2)(dense_1)
    dense_2 = Dense(hidden_units, activation='relu')(dropout)

    predictions = Dense(1, activation='sigmoid')(dense_2)

    clf = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

    clf.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return clf


def build_comparable_model(word_index, embedding_matrix):
    tweet_a = Input(shape=(tweet_length,))
    tweet_b = Input(shape=(tweet_length,))

    emb = Embedding(
        input_dim=len(word_index) + 1, output_dim=embedding_dim, trainable=True,
        input_length=tweet_length, mask_zero=True, weights=[embedding_matrix])
    gru = GRU(hidden_units, return_sequences=True)
    average_pooling = keras.layers.Lambda(lambda x_in: K.mean(x_in, axis=1))

    encoded_a = average_pooling(gru(emb(tweet_a)))
    encoded_b = average_pooling(gru(emb(tweet_b)))

    cosine = keras.layers.dot([encoded_a, encoded_b], axes=-1, normalize=True)

    predictions = Dense(1, activation='sigmoid')(cosine)

    clf = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

    clf.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return clf
