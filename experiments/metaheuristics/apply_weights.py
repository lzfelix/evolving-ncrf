import json
import argparse

import numpy as np
from sklearn import metrics

from py_src.utils import get_weights
from py_src.utils import read_labels
from py_src.utils import read_all_predictions
from py_src.utils import linear_combination
from py_src.utils import show_basic_stats

# JSON key names
KEY_PRED_FILES = 'pred_files'
AC_KEY_WEIGHTS = 'ac_weights_file'
F1_KEY_WEIGHTS = 'f1_weights_file'

GA_KEY = 'ga_weights'
GP_KEY = 'gp_weights'

ACC_KEY = 'ac_weights_file'
MF1_KEY = 'f1_weights_file'

KEY_LABELS = 'labels_file'

JSON_SPECS_FILEPATH = '../specs.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='Applies evolutionary weights to each of the candidates forming the ensamble. ' +
                                       'Run with -h for help')
    parser.add_argument('model_name', help='Dataset used to train the model. Format: [dataset_model]. '+
                        'datasets=(ds1|ds3); model=(softmax_crf)', type=str)
    parser.add_argument('split', help='Split used to compute the metrics (dev|tst).')
    parser.add_argument('metric', help='Either accuracy (acc) or macro_f1 (f1)')
    parser.add_argument('algo', help='Ensembling type (ga|gp)', default='ga')

    args          = parser.parse_args()
    model_name    = args.model_name
    dataset_split = args.split.lower()
    metric        = args.metric.lower()
    algo          = args.algo.lower()

    if algo == 'ga':
        evolution_key = GA_KEY
    elif algo == 'gp':
        evolution_key = GP_KEY
    else:
        raise RuntimeError('Valid values for algo are either "ga" or "gp".')

    if metric == 'acc':
        fitness_fn_key = ACC_KEY
    elif metric == 'f1':
        fitness_fn_key = MF1_KEY
    else:
        raise RuntimeError('Valid values for metric are either "acc" or "f1".')

    with open(JSON_SPECS_FILEPATH) as dfile:
        all_specs = json.load(dfile)

    weights_filepath          = all_specs[model_name][evolution_key][fitness_fn_key]
    all_predictions_filepaths = all_specs[model_name][dataset_split][KEY_PRED_FILES]
    labels_filepath           = all_specs[model_name][dataset_split][KEY_LABELS]

    weights_filepath = '../' + weights_filepath

    print(f'Model name       : {model_name}')
    print(f'Algorithm        : {algo}')
    print(f'Dataset split    : {dataset_split}')
    print(f'Metric           : {metric}')
    print(f'Using weights at : {weights_filepath}')
    print(f'Labels filepath  : {labels_filepath}')

    # Reading samples and labels file
    all_predictions_filepaths = ['../' + filepath for filepath in all_predictions_filepaths]
    n_samples, n_classes, all_predictions = read_all_predictions(all_predictions_filepaths)
    l_samples, l_classes, y_true = read_labels('../' + labels_filepath)
    
    # Sanity check
    if l_samples != n_samples or l_classes != n_classes:
        print(f'n_samples = {n_samples}')
        print(f'n_classes = {n_classes}')
        print(f'l_samples = {l_samples}')
        print(f'l_classes = {l_classes}')
        raise RuntimeError('[ERROR] Amount of labels differs from the amount of predictions')

    # Reading genetic weights
    all_candidates_weights = np.asarray(get_weights(weights_filepath))
    
    # Computing metrics for each ensamble
    all_accs = list()
    all_f1s  = list()
    for candidates_weights in all_candidates_weights:
        ensamble_predictions = linear_combination(candidates_weights, all_predictions)
        y_pred = np.argmax(ensamble_predictions, -1)

        all_accs.append(metrics.accuracy_score(y_true, y_pred))
        all_f1s.append(metrics.f1_score(y_true, y_pred, average='macro'))

    print('Accuracies:')
    print('-----------')
    show_basic_stats(all_accs)

    print('F1:')
    print('---')
    show_basic_stats(all_f1s)
