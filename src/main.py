from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn import metrics
import numpy as np
import argparse
from enum import Enum
import pre_process
import dataset_loader
import siamese_models
import variations


class Variation(Enum):
    none = 'none'
    fasttext = 'fasttext'
    meta = 'meta'
    idf = 'idf'

    def __str__(self):
        return self.value


callbacks = [
    EarlyStopping(min_delta=1e-3, patience=2, verbose=1),
    ReduceLROnPlateau(factor=2e-2, patience=1, verbose=1),
    CSVLogger("training-log.csv"),
]


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pyspark job for Places duplicate detection"
    )

    parser.add_argument(
        "tweets_path",
        type=str,
        help='path from where the raw tweets should be read')
    parser.add_argument(
        "embeddings_path",
        type=str,
        help='path from where the embeddings should be read')
    parser.add_argument(
        "train_path",
        type=str,
        help='path from where the training set pairs should be read')
    parser.add_argument(
        "val_path",
        type=str,
        help='path from where the validation set pairs should be read')
    parser.add_argument(
        "test_path",
        type=str,
        help='path from where the test set pairs should be read')
    parser.add_argument(
        "--variation",
        default=Variation.none,
        type=Variation,
        choices=list(Variation),
        help='model variation. Can be none, fasttext, meta, or idf')

    return parser


def get_matrix_from_variation(variation, corpus, word_index, base_matrix):
    if variation is Variation.none:
        return base_matrix
    elif variation is Variation.fasttext:
        ft = variations.build_fasttext(corpus)
        return variations.build_fasttext_matrix(ft, word_index)
    elif variation is Variation.meta:
        ft = variations.build_fasttext(corpus)
        ft_matrix = variations.build_fasttext_matrix(ft, word_index)
        return variations.build_meta_matrix(base_matrix, ft_matrix, word_index)
    elif variation is Variation.idf:
        return variations.build_idf_matrix(corpus, base_matrix, word_index)
    else:
        return base_matrix


def get_metrics(clf, test_set):
    y_pred = [1 if x > 0.5 else 0 for x in clf.predict(
        [test_set[0], test_set[1]])]
    y_test = list(map(int, test_set[2]))
    return (np.array(y_pred) == np.array(y_test)).sum() / len(y_pred), \
        metrics.precision_recall_fscore_support(y_test, y_pred)


def main():
    args = get_parser().parse_args()
    print("Using variation: " + str(args.variation))
    word_index, embeddings_matrix, corpus = pre_process.pre_processing_pipeline(
        args.tweets_path, args.embeddings_path)

    print("Loading and pre-processing data...")
    train = dataset_loader.read_tabular_dataset(args.train_path)
    val = dataset_loader.read_tabular_dataset(args.val_path)
    test = dataset_loader.read_tabular_dataset(args.test_path)
    train_pre_processed = pre_process.pre_process_pairs(train, word_index)
    val_pre_processed = pre_process.pre_process_pairs(val, word_index)
    test_pre_processed = pre_process.pre_process_pairs(test, word_index)

    chosen_matrix = get_matrix_from_variation(
        args.variation, corpus, word_index, embeddings_matrix)

    print("Compiling models...")
    ours = siamese_models.build_model(word_index, chosen_matrix)
    comparable = siamese_models.build_comparable_model(
        word_index, chosen_matrix)

    print("Training our model...")
    ours.fit(
        [train_pre_processed[0],
         train_pre_processed[1]],
        train_pre_processed[2],
        validation_data=([val_pre_processed[0],
                          val_pre_processed[1]],
                         val_pre_processed[2]),
        epochs=10, batch_size=32, callbacks=callbacks)

    print("Metrics for our model:\n")
    get_metrics(ours, test_pre_processed)

    import gc
    del ours
    gc.collect()

    print("Training comparable model...")
    comparable.fit(
        [train_pre_processed[0],
         train_pre_processed[1]],
        train_pre_processed[2],
        validation_data=([val_pre_processed[0],
                          val_pre_processed[1]],
                         val_pre_processed[2]),
        epochs=10, batch_size=32, callbacks=callbacks)

    print("Metrics for comparable model:\n")
    get_metrics(comparable, test_pre_processed)


if __name__ == "__main__":
    main()
