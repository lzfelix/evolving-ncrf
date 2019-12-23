#ifndef __FILE_UTILS_H__
#define __FILE_UTILS_H__

#include <stdio.h>
#include "matrix.h"

FILE* read_file(char *filepath) {
    /* Tries to read a file and aborts the program in case of failure.*/

    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "Could not read file %s.\n", filepath);
        exit(-1);
    }
    return fp;
}

int* load_labels(char *filepath, int n_samples, int n_labels) {
    /* Loads the ground truth labels from file with format
    
        n_labels n_classes
        label_1
        label_2
        ...
        label_{n_labels}
    */

    int new_samples, new_labels;
    FILE *fp = read_file(filepath);
    fscanf(fp, "%d %d", &new_samples, &new_labels);

    int *labels = malloc(sizeof(int) * new_samples);
    for (int i = 0; i < n_samples; i++) {
        fscanf(fp, "%d", &labels[i]);
    }
    fclose(fp);

    if (new_samples != n_samples || new_labels != n_labels) {
        fprintf(stderr, "Amount of ground truth labels (%d) differ from amount "
                "of predictions (%d)\n", new_samples, n_samples);
        exit(-1);
    }

    return labels;
}

double** load_file(char *filepath, int *n_samples, int *n_classes) {
    /* Opens a prediction file in the following format:

        n_predictions n_classes accuracy f1
        pred_1
        pred_2
        ...
        pred_{n_predictions}

    Then returns a 1-hot encoded matrix with shape (n_predictions, n_classes).
    After runing this function, *n_samples contains the amount of rows in the
    matrix.
    */

    FILE *fp = read_file(filepath);

    // Getting amount of samples and classes
    float _;
    fscanf(fp, "%d %d %f %f", n_samples, n_classes, &_, &_);

    // Reading predictions as one-hot encoding
    int class_index;
    double **one_hot = allocate_matrix(*n_samples, *n_classes);
    for (int i = 0; i < *n_samples; i++) {
        fscanf(fp, "%d", &class_index);
        one_hot[i][class_index] = 1;
    }

    fprintf(stdout, "Read %d samples from file '%s'\n", *n_samples, filepath);
    fclose(fp);
    return one_hot;
}

double** load_prob_distr(char *filepath, int *n_samples, int *n_classes) {
    /* Opens a prediction file in the following format:
    
        n_predictions n_classes accuracy f1
        p(y_0 | x_0) p(y_1 | x_0) ... p(y_{n_classes} | x_0)
        p(y_0 | x_1) p(y_1 | x_1) ... p(y_{n_classes} | x_1)
        ...
        p(y_0 | x_{n_predictions}) ... p(y_{n_classes} | x_{n_predictions})
    */

    float _;
    FILE *fp = read_file(filepath);

    // Getting amount of samples and classes
    fscanf(fp, "%d %d %f %f", n_samples, n_classes, &_, &_);

    double **predictions = allocate_matrix(*n_samples, *n_classes);
    for (int row = 0; row < *n_samples; row++) {
        for (int class = 0; class < *n_classes; class++) {
            fscanf(fp, "%lf", &predictions[row][class]);
        }
    }

    fprintf(stdout, "Read %d samples from file '%s'\n", *n_samples, filepath);
    fclose(fp);
    return predictions;
}

double*** load_multiple_predictions(char *filepaths[], int n_candidates, int *n_samples, int *n_classes, char is_hot) {
    int new_samples, new_classes;
    double*** all_predictions = malloc(sizeof(double**) * n_candidates);

    fprintf(stdout, "There are %d candidates.\n", n_candidates); 
    for (int i = 0; i < n_candidates; i++) {
        if (is_hot) {
            all_predictions[i] = load_file(filepaths[i], &new_samples, &new_classes);
        }
        else {
            all_predictions[i] = load_prob_distr(filepaths[i], &new_samples, &new_classes);
        }

        if (i == 0) {
            *n_samples = new_samples;
            *n_classes = new_classes;

            fprintf(stdout, "Found %d samples with %d classes\n",  *n_samples, *n_classes);
        }
        else if (new_samples != *n_samples || *n_classes != new_classes) {
            fprintf(stderr, "Amount of samples read from file %d (%d) is different " 
                    "from previous amount (%d).", i, new_samples, *n_samples);
            exit(-1);
        }
    }

    return all_predictions;
}

#endif
