import re
import os
import argparse

import numpy as np


def find_model_name(lines):
    # Finding any line with the model path
    path_line = None
    for line in lines:
        if 'saving model to' in line:
            path_line = line
            break

    # Isolating the path
    path_line = path_line[path_line.find(' to ') + 4:]
    model_name = os.path.basename(path_line.strip())
    return model_name


def analyse_block(lines):
    METRICS_PATTERN = re.compile(r'\d\.\d*')
    VAL_LOSS_INDEX = 2

    loss_lines = [line.strip() for line in lines if 'val_loss:' in line]
    validation_losses = []
    for loss_line in loss_lines:
        epoch_metrics = re.findall(METRICS_PATTERN, loss_line)
        epoch_validation = float(epoch_metrics[VAL_LOSS_INDEX])
        validation_losses.append(epoch_validation)
    return validation_losses


def divide_losses_into_blocks(val_losses, lr_intervals):
    lower_bound = 0
    best_indices = list()
    best_values = list()
    
    for upper_bound in lr_intervals:
        losses = val_losses[lower_bound:upper_bound]

        min_index = np.argmin(losses) + 1
        min_value = np.min(losses)
        
        best_indices.append(lower_bound + min_index)
        best_values.append(min_value)

        lower_bound = upper_bound
        
    return best_indices, best_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='Script to identify in which epochs the model achieved the lowest ' +
                                     'validation error within a learning rate cycle (cycles = [2, 6, 14, 30]). ' +
                                     'These filenames are available at ../metaheuristics/README.md. ' +
                                     'Run with -h for details.')
    parser.add_argument('filepath', help='Path to logs saved from training models multiple times. ' +
                        'If you do "python3 full_model_store_partials.py > ./output.txt 2>&1" you should ' +
                        'supply "../experiments/output.txt" here', type=str)
    args = parser.parse_args()

    print(f'Loading results from {args.filepath}\n')    
    with open(args.filepath) as dfile:
        all_lines = dfile.readlines()

    lr_intervals = [2, 6, 14, 30]
    start = 0
    end = 0
    block_no = 0

    all_indices = list()
    all_names = list()

    for lineno, line in enumerate(all_lines):
        if re.match(r'--------------- << Training model \[\d+\/15\] >> ---------------', line):
            start = end
            end = lineno
            if block_no > 0:
                model_name = find_model_name(all_lines[start:end])
                validation_losses = analyse_block(all_lines[start:end])
                best_indices, best_losses = divide_losses_into_blocks(validation_losses, lr_intervals)

                this_training_indices = list()
                for index, loss in zip(best_indices, best_losses):
                    this_training_indices.append(index)

                all_indices.append(this_training_indices)
                all_names.append(model_name)
            block_no += 1

    for model_name, min_indices in zip(all_names, all_indices):
        print('{}\t{}'.format(model_name, min_indices))
