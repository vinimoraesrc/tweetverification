from sklearn.svm import LinearSVC as SVM
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import cosine
import numpy as np


def calc_similarity_unit(idx1, idx2, matrix):
    vec1 = matrix[idx1]
    vec2 = matrix[idx2]
    return cosine(vec1, vec2)


def calc_similarity_vector(matrix, idxs1, idxs2):
    return [calc_similarity_unit(idxs1[i], idxs2[i], matrix) for i in range(len(idxs1))]


def get_vector_cosine(matrix, pair_a, pair_b):
    return [calc_similarity_vector(matrix, pair_a[i], pair_b[i]) for i in range(len(pair_a))]


def average_embed(matrix, idxs):
    vec = [np.array(matrix[idx]) for idx in idxs]
    vec_sum = np.sum(np.array(vec), axis=0)
    return vec_sum/len(idxs)


def calc_similarity_metric(matrix, idxs1, idxs2):
    vec1 = average_embed(matrix, idxs1)
    vec2 = average_embed(matrix, idxs2)
    sim = cosine(vec1, vec2)
    return sim


def get_single_cosine(matrix, pair_a, pair_b):
    return [[calc_similarity_metric(matrix, pair_a[i], pair_b[i])] for i in range(len(pair_a))]


def calc_concat_vector(matrix, idxs1, idxs2):
    vec1 = average_embed(matrix, idxs1)
    vec2 = average_embed(matrix, idxs2)
    return np.concatenate((vec1, vec2), axis=0)


def get_concat_vector(matrix, pair_a, pair_b):
    return [calc_concat_vector(matrix, pair_a[i], pair_b[i]) for i in range(len(pair_a))]


def clean(vec, idxs):
    return [vec[i] for i in range(len(vec)) if i not in idxs]


def remove_nans(vec):
    nan_idxs = set([i for i in range(len(vec)) if np.isnan(vec[i])])
    return clean(vec, nan_idxs)


def get_svm():
    return SVM(dual=False)
