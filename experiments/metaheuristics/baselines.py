import json
import argparse
from pathlib import Path

import numpy as np
from sklearn import metrics

from py_src.utils import read_all_predictions
from py_src.utils import read_labels
from py_src.utils import linear_combination

JSON_SPECS_FILEPATH = '../specs.json'
KEY_PRED_FILES = 'pred_files'
KEY_LABELS = 'labels_file'


def _combine_predict(weights, all_predictions, y_true):
    predictions = linear_combination(weights, all_predictions)
    y_pred = np.argmax(predictions, axis=-1)
    
    acc = metrics.accuracy_score(y_true, y_pred)
    mf1 = metrics.f1_score(y_true, y_pred, average='macro')
    return acc, mf1


def uniform_combination(y_true, all_predictions):
    n_candidates = len(all_predictions)

    return _combine_predict(
        weights=np.ones(n_candidates) / n_candidates,
        all_predictions=all_predictions,
        y_true=y_true
    )


def random_combination(y_true, all_predictions):
    n_candidates = len(all_predictions)
    
    random_weights = np.random.random(n_candidates)
    random_weights /= random_weights.sum().item()
    return _combine_predict(
        weights=random_weights,
        all_predictions=all_predictions,
        y_true=y_true
    )


def final_combination(y_true, all_predictions, all_predictions_filepaths):
    
    def get_model_name(filepath):
        basename = Path(filepath).stem
        basename = basename[:basename.find('=')]
        return basename
    
    def get_epoch(filepath):
        dot_index = filepath.rfind('.')
        equals_index = filepath.find('=')
        
        return int(filepath[equals_index+1:dot_index])

    def split_by_names(filepaths):
        common = []
        cache = [filepaths[0]]
        previous = get_model_name(filepaths[0])

        for filepath in filepaths[1:]:
            cur_base = get_model_name(filepath)
            if cur_base != previous:
                previous = cur_base
                common.append(cache)
                cache = []
            cache.append(filepath)

        common.append(cache)
        return common
    
    # Initially there is a list of candidate filepaths. Split these paths
    # into smaller lists, each containing paths to each candidate checkpoint
    training_paths = split_by_names(all_predictions_filepaths)
    
    # For each candidate get only their corresponding epoch number
    training_paths_epochs = [
        [get_epoch(filepath) for filepath in path]
        for path in training_paths
    ]
    
    # Figure out what is the index of the last checkpoint (presumably
    # the best model)
    indices = [np.argmax(epochs) for epochs in training_paths_epochs]
    
    # Now it is possible to combine these guys
    return uniform_combination(
        y_true,
        [all_predictions[index] for index in indices]
    )


def _print_stats(name, acc, mf1):
    print(name)
    print('\t Accuracy: {:4.4}'.format(acc))
    print('\t Macro-F1: {:4.4}'.format(mf1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='Computes diferent ensemble baselines. Run with -h for help')
    parser.add_argument('model_name', help='Dataset used to train the model. Format: [dataset_model]. '+
                        'datasets=(dsC|dsD); model=(softmax|crf). Ex: dsC_softmax', type=str)

    args = parser.parse_args()
    model_name = args.model_name
    dataset_split = 'tst'

    all_specs = json.load(Path(JSON_SPECS_FILEPATH).open('r'))
    all_predictions_filepaths = all_specs[model_name][dataset_split][KEY_PRED_FILES]
    labels_filepath           = all_specs[model_name][dataset_split][KEY_LABELS]

    print(f'Model name       : {model_name}')
    print(f'Dataset split    : {dataset_split}')
    print(f'Labels filepath  : {labels_filepath}')

    all_predictions_filepaths = ['../' + filepath for filepath in all_predictions_filepaths]
    labels_filepath = '../' + labels_filepath
    n_samples, n_classes, all_predictions = read_all_predictions(all_predictions_filepaths)
    l_samples, l_classes, y_true = read_labels(labels_filepath)

    # Sanity check
    if l_samples != n_samples or l_classes != n_classes:
        print(f'n_samples = {n_samples}')
        print(f'n_classes = {n_classes}')
        print(f'l_samples = {l_samples}')
        print(f'l_classes = {l_classes}')
        raise RuntimeError('[ERROR] Amount of labels differs from the amount of predictions')

    acc, mf1 = uniform_combination(y_true, all_predictions)
    _print_stats('Uniform combination', acc, mf1)

    acc, mf1 = random_combination(y_true, all_predictions)
    _print_stats('Random combination', acc, mf1)
    
    acc, mf1 = final_combination(y_true, all_predictions, all_predictions_filepaths)
    _print_stats('Best candidates combination', acc, mf1)
