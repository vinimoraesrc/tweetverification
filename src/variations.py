from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import FastText
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import df2idf
import numpy as np


def build_fasttext(corpus):
    path = get_tmpfile("/content/fasttext.model")
    model_ft = FastText(corpus, size=50, window=5,
                        min_count=1, workers=2, iter=15)
    model_ft.save(path)
    path_kv_ft = get_tmpfile("/content/fasttext.kv")
    model_ft.wv.save(path_kv_ft)
    return model_ft


def build_fasttext_matrix(model, word_index):
    gensim_embeddings_index = {k: model.wv[k]
                               for k in model.wv.index2word}
    gensim_embedding_matrix = np.zeros((len(word_index) + 1, 50))
    for word, index in word_index.items():
        embedding_vector = gensim_embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            gensim_embedding_matrix[index] = embedding_vector
    return gensim_embedding_matrix


def build_meta_matrix(base_matrix, support_matrix, word_index):
    meta_embedding_matrix = base_matrix.copy()
    for word, index in word_index.items():
        if (meta_embedding_matrix[index] is None):
            meta_embedding_matrix[index] = base_matrix[index]
    return meta_embedding_matrix


def build_idf_matrix(corpus, base_matrix, word_index):
    dct = Dictionary(corpus)
    corpus_matrix = [dct.doc2bow(line) for line in corpus]

    word_dfs = {}
    for tweet in corpus:
        curr = {}
        for word in tweet:
            if word not in curr:
                curr[word] = 1
            word_dfs[word] = word_dfs.get(word, 0) + 1

    word_idfs = [[x, df2idf(y, len(corpus))]
                 for x, y in word_dfs.items()]
    word_idfs = dict(word_idfs)

    weighted_embedding_matrix = base_matrix.copy()
    for word, index in word_index.items():
        weight = word_idfs[word]
        weighted_embedding_matrix[index] = [weight * x
                                            for x in weighted_embedding_matrix[index]]
    return weighted_embedding_matrix
