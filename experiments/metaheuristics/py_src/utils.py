import re
import numpy as np


def get_weights(filepath):
    """Reads the logs output by optimization and gets only the weights"""

    all_weights = list()
    for lineno, line in enumerate(open(filepath)):
        # The line with weights has format w = [0.1234, ...., (no closing bracket
        if line[0] == '#':
            continue

        if line[0] == 'w':
            weights = re.findall(r'\d\.\d*', line)
            all_weights.append(list(map(float, weights)))
    return all_weights


def read_labels(filepath):
    """Reads the labels file and returns a 1D matrix with the labels for each sample."""

    predictions = list()
    for lineno, line in enumerate(open(filepath)):    
        if lineno == 0:
            n_samples, n_classes = line.split()
        else:
            predictions.append(int(line))
    return int(n_samples), int(n_classes), np.asarray(predictions)


def read_predictions_file(filepath):
    """Reads a prediction file in the format below and returns a `(n_samples, n_classes) matrix,
    either 1-hot encoded (if the file has Format 1) or with the probability distribution for each
    sample (if the file has Format 2).
    
    Format 1:
    ```
        n_samples n_classes accuracy f1
        prediction_0
        prediction_1
        ...
        prediction_{n_samples - 1}
    ```
    
    Format 2:
    ```
        n_samples n_classes accuracy f1
        p(y_0 | x_0) p(y_1 | x_0) ... p(y_{n_classes - 1}  | x_0)
        p(y_0 | x_1) p(y_1 | x_1) ... p(y_{n_classes - 1}  | x_1)
        ...
        p(y_0 | x_{n_samples - 1}) p(y_1 | x_{n_samples - 1}) ... p(y_{n_classes - 1}  | x_{n_samples - 1})
    ```
    """

    y_pred = None
    for lineno, line in enumerate(open(filepath)):
        if lineno == 0:
            n_samples, n_classes, _, _ = line.split()
            n_samples = int(n_samples)
            n_classes = int(n_classes)

            y_pred = np.zeros(shape=(n_samples, n_classes))
        else:
            elements = line.split()
            if len(elements) == 1:
                # This is a 1-hot-encoded file
                index = int(elements[0])
                y_pred[lineno - 1, index] = 1
            else:
                # This is a line with probabilities distribution. Populate the whoe line
                elements = np.asarray(list(map(float, elements)))
                y_pred[lineno - 1] = elements

    return n_samples, n_classes, y_pred


def read_all_predictions(filepaths):
    all_predictions = list()
    current_samples = 0
    n_samples = 0
    
    for filepath in filepaths:
        n_samples, n_classes, predictions = read_predictions_file(filepath)
        if current_samples == 0:
            current_samples = n_samples
            current_classes = n_classes
        elif current_samples != n_samples or current_classes != n_classes:
            raise RuntimeError(f'[ERROR] Amount of samples/classes for "{filepath}" differ from the remaining')

        all_predictions.append(predictions)
    return n_samples, n_classes, all_predictions


def linear_combination(weights, arrays):
    """Computes $\sum weight[i] * arrays[i]$."""

    buffer = np.zeros_like(arrays[0])
    weights_sum = weights.sum().item() # <<<<< (!)
    for weight, array in zip(weights, arrays):
        buffer += weight / weights_sum * array

    return buffer


def show_basic_stats(array):
    print('\t - avg: {:.4}'.format(np.average(array) * 100))
    print('\t - std: {:.4}'.format(np.std(array) * 100))
    print('\t - max: {:.4}'.format(np.max(array) * 100))
    print('\t - min: {:.4}'.format(np.min(array) * 100))
    print('\n{}'.format(array))
