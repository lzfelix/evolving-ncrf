import os
import sys
import tqdm
import json
import argparse
import functools
from pathlib import Path
if '../' not in sys.path:
    sys.path.append('../')

import keras
import numpy as np

from gensim.models import KeyedVectors
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from sklearn.metrics import accuracy_score, f1_score

from scripts.layers import AttentionLayer
from scripts import loader
from scripts import seq_utils
from scripts import preprocessing as pre


# JSON keys
KEY_CHECKPOINTS = 'checkpoints'
KEY_MODELS_DIR = 'models_folder'
KEY_DETAILS = 'details'
KEY_MODE = 'mode'
KEY_ALL_FILES = 'all_files'
KEY_EMBEDDINGS = 'embeddings'
KEY_TRN = 'trn_files'
KEY_DEV = 'dev_files'
KEY_TST = 'tst_files'
KEY_BDP_SEQ_SIZE = "bdp_seq_size"
KEY_SENT_LEN = "max_sent_len"
KEY_BDO_SEQ_OVER = "bdp_seq_overlap"
KEY_OTHER_CLASSES = "other_classes"
KEY_CLASS_MAP = "class_map"
KEY_OTHER_CLASSES = "other_classes"
KEY_CLASS_MAP = "class_map"

JSON_MODELS_FILEPATH = '../specs.json'

# Model constants
VOCAB_SIZE = 2000


def get_parameters() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage='Performs predictions on the dataset using using CLR minima points')
    parser.add_argument('model_name', help='Name of the model used for prediction', type=str)
    parser.add_argument('dataset_name', help='Name of the dataset [dsC/dsD]', type=str)
    parser.add_argument('destination_folder', help='Where the predictions are going to be stored', type=str)
    return parser.parse_args()


def load_json(filepath):
    with open(filepath) as dfile:
        return json.load(dfile)

    
def load_model(filepath):
    print(f'Loading model {filepath}')
    sys.stdout.flush()
    return keras.models.load_model(filepath,
                                   custom_objects={'CRF':CRF,
                                                   'crf_loss': crf_loss,
                                                   'crf_viterbi_accuracy': crf_viterbi_accuracy,
                                                   'AttentionLayer': AttentionLayer})


def compute_predictions(filepath, X):
    """Gets the most likely label regardless hard or soft prediction."""
    model = load_model(filepath)
    return model.predict(X).argmax(-1).flatten()


def compute_metrics(filepath, X, y):
    y_hat = compute_predictions(filepath, X)
    y_flat = y.flatten()

    non_masked_indices = (y_flat != 0)
    y_ground = y_flat[non_masked_indices]
    y_hat = y_hat[non_masked_indices]
    
    acc = accuracy_score(y_ground, y_hat)
    f1  = f1_score(y_ground, y_hat, average='macro')
    return y_hat, acc, f1


def compute_pred_distr(filepath, n_classes, X, y):
    """Loads a model from @filepath, computes the distribution of probabilities for @X
    and returns this distribution, accuracy and F1 score."""

    # Computing label probability distribution (not 1-hot)
    model = load_model(filepath)
    y_hat_distr = model.predict(X).reshape((-1, n_classes))

    # Flattening ground truth and getting non-maked samples, y is always 1-hot
    y_flat = y.flatten()
    non_masked_indices = (y_flat != 0)

    # Getting non-masked labels and probabilities
    y_ground = y_flat[non_masked_indices]
    y_hat_distr = y_hat_distr[non_masked_indices]

    # To compute the metrics, the argmax is needed instead
    y_hat_hot = np.argmax(y_hat_distr, -1)
    acc = accuracy_score(y_ground, y_hat_hot)
    f1 = f1_score(y_ground, y_hat_hot, average='macro')

    return y_hat_distr, acc, f1


def get_model_name(model_filepath):
    basename = os.path.basename(model_filepath)
    return basename[:basename.rfind('.')]


def dump_predictions(destination_folder, model_name, n_classes, predictions, accuracy, f1):
    # Figuring out where store the file
    destination_folder = Path(destination_folder)
    logs_file = destination_folder/(model_name + '.txt')
    logs_file.open('w')

    n_samples = len(predictions)
    
    if isinstance(predictions[0], np.ndarray):
        # if predictions the 1st prediction is a list of ndarrays, then all of them 
        # are lists of probability distributions. Need to format this properly
        formated_predictions = [
            ' '.join(map(str, prediction))
            for prediction in predictions
        ]
        predictions = formated_predictions
    predictions_as_str = '\n'.join(map(str, predictions))

    # Dumping to file
    logs_file.write_text('{} {} {} {} \n{}'.format(n_samples, n_classes, accuracy, f1, predictions_as_str))


def dump_labels(destination_folder, n_classes, y, is_test_label):
    y_flat = y.flatten()
    non_masked_indices = (y_flat != 0)
    y_ground = y_flat[non_masked_indices]
    y_ground_as_str = '\n'.join(map(str, y_ground))
    
    n_samples = len(y_ground)

    suffix = 'test_labels.txt' if is_test_label else 'labels.txt'
    path = Path(destination_folder)/suffix
    path.write_text('{} {}\n{}'.format(n_samples, n_classes, y_ground_as_str))

    
if __name__ == '__main__':
    args = get_parameters()
    MODEL_NAME = args.model_name
    DESTINATION_DIR = args.destination_folder
    DATASET_NAME = MODEL_NAME.split('_')[0]
    
    is_one_hot = 'crf' in MODEL_NAME
    specs = load_json(JSON_MODELS_FILEPATH)[MODEL_NAME]
    data = loader.load_specs(DATASET_NAME)

    models_dir = specs[KEY_MODELS_DIR]
    candidates = specs[KEY_CHECKPOINTS]
    sequence_hyperparams = specs[KEY_DETAILS]

    print('Selected candidates:')
    for checkpoint, epochs in candidates.items():
        print(f'\t - {checkpoint}\t{epochs}')

    print(f'Models dir: {models_dir}')
    print(f'[!] Is 1-hot   : {is_one_hot}')
    print(f'[!] Dataset    : {DATASET_NAME}')
    print(f'[!] Embeddings : {specs[KEY_EMBEDDINGS]}')
    sys.stdout.flush()

    # Building the vocabulary
    X, _ = loader.load_all_excel('../../data/' + data[KEY_ALL_FILES], merge=True, mode=data[KEY_MODE])
    X = pre.advanced_pre_no_numbers(X, None)

    # Building word embedding matrix
    embeddings = KeyedVectors.load(specs[KEY_EMBEDDINGS])
    embeddings.init_sims(replace=True)
    E, word2index, missing = loader.load_embeddings_matrix(embeddings, X, top_k=VOCAB_SIZE)
    vocab = set(word2index.keys())
    print(f'{missing} words have no embedding.')
    
    # Loading data
    pload = functools.partial(
        seq_utils.load_segments, 
        word2index=word2index,
        sentence_len=sequence_hyperparams[KEY_SENT_LEN],
        segment_size=sequence_hyperparams[KEY_BDP_SEQ_SIZE],
        overlap=sequence_hyperparams[KEY_BDO_SEQ_OVER],
        vocab=vocab,
        other_classes=data[KEY_OTHER_CLASSES],
        classmap=data[KEY_CLASS_MAP],
        mode=data[KEY_MODE],
        shuffle=False
    )

    trn_x, trn_y, lencoder = pload(files=data[KEY_TRN])
    dev_x, dev_y, _        = pload(files=data[KEY_DEV], lencoder=lencoder)
    tst_x, tst_y, _        = pload(files=data[KEY_TST], lencoder=lencoder)
    
    n_classes = len(lencoder.classes_)

    # Cleaning up some stuff
    print('Input features matrix shape: ', trn_x.shape)
    print('Amount of classes (+ PAD class): ', len(lencoder.classes_))

    del X
    del missing
    
    # Assembling candidates filepaths    
    selected_filepaths = list()
    for model_filename, selected_epochs in candidates.items():
        # (...).append(../experiments/blah/model_xxx_epoch=12.h5)
        selected_filepaths.extend([
            os.path.join(models_dir, model_filename.format(epoch))
            for epoch in selected_epochs
        ])
    print(selected_filepaths)

    # Loading model candidates and computing the metrics
    all_preds = list()
    all_accs = list()
    all_f1s = list()

    for model_filepaths in tqdm.tqdm(selected_filepaths):
        if is_one_hot:
            y_hat, acc, f1 = compute_metrics(model_filepaths, dev_x, dev_y)
        else:
            y_hat, acc, f1 = compute_pred_distr(model_filepaths, n_classes, dev_x, dev_y)

        all_preds.append(y_hat)
        all_accs.append(acc)
        all_f1s.append(f1)
        print(acc)
    
    # Storing the prediction results
    for i in range(len(selected_filepaths)):
        dump_predictions(
            DESTINATION_DIR,
            get_model_name(selected_filepaths[i]),
            len(lencoder.classes_),
            all_preds[i],
            all_accs[i],
            all_f1s[i]
        )

    dump_labels(DESTINATION_DIR, len(lencoder.classes_), dev_y, is_test_label=False)
    
    # Predicting for test set and dumping labels as well
    all_preds = list()
    all_accs = list()
    all_f1s = list()

    for model_filepaths in tqdm.tqdm(selected_filepaths):
        if is_one_hot:
            y_hat, acc, f1 = compute_metrics(model_filepaths, tst_x, tst_y)
        else:
            y_hat, acc, f1 = compute_pred_distr(model_filepaths, n_classes, tst_x, tst_y)

        all_preds.append(y_hat)
        all_accs.append(acc)
        all_f1s.append(f1)

    for i in range(len(selected_filepaths)):
        dump_predictions(
            DESTINATION_DIR,
            'test_' + get_model_name(selected_filepaths[i]),
            len(lencoder.classes_),
            all_preds[i],
            all_accs[i],
            all_f1s[i]
        )

    dump_labels(DESTINATION_DIR, len(lencoder.classes_), tst_y, is_test_label=True)
