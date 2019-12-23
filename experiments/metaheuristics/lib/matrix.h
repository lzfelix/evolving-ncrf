#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int** allocate_int_matrix(int rows, int cols) {
    int **matrix = malloc(sizeof(*matrix) * rows);
    for (int i = 0; i < cols; i++) {
        matrix[i] = calloc(cols, sizeof(matrix[i]));
    }
    return matrix;
}

double** allocate_matrix(int n_samples, int n_classes) {
    /* Creates a (n_samples, n_classes) integer matrix. */

    double **matrix = malloc(sizeof(double*) * n_samples);
    for (int i = 0; i < n_samples; i++) {
        matrix[i] = calloc(n_classes, sizeof(double));
    }

    return matrix;
}

void destroy_matrix(double **matrix, int n_samples) {
    /* Iterativelly frees the memory blocks used to hold **matrix. */

    for (int i = 0; i < n_samples; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void destroy_int_matrix(int **matrix, int n_rows) {
    for (int i = 0; i < n_rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void destroy_matrix_list(double ***list, int n_elements, int n_samples) {
    for (int j = 0; j < n_elements; j++) {
        destroy_matrix(list[j], n_samples);
    }
    free(list);
}

void print_matrix(double **matrix, int n_samples, int n_classes) {
    /* Prints the contents from **matrix. Can print just the first n_samples
    rows and n_classes cols. */

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_classes; j++)
            printf("%lf   ", matrix[i][j]);
        printf("\n");
    }
}

void print_int_matrix(int **matrix, int n_samples, int n_classes) {
    /* Prints the contents from **matrix. Can print just the first n_samples
    rows and n_classes cols. */

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_classes; j++)
            printf("%d   ", matrix[i][j]);
        printf("\n");
    }
}

void print_int_array(int* arr, int len) {
    for (int i = 0; i < len; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

double **mult_add_store(double scalar, double **predictions, double **dest, int rows, int cols) {
    /* Computes dest += scalar * (predictions[i][j]) */

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i][j] += scalar * predictions[i][j];
        }
    }
    return dest;
}

double **restore(double **buffer, int rows, int cols) {
    /* Sets all rows from the buffer matrix to zero. */

    for (int i = 0; i < rows; i++) {
        buffer[i] = memset(buffer[i], 0, sizeof(double) * cols);
    }
    return buffer;
}

double **scale_matrix(double scalar, double **predictions, int rows, int cols) {
    /* Multiplies (not in place) all values from **predictions by scalar. */

    double **scaled_copy = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            scaled_copy[i][j] = predictions[i][j] * scalar;
        }
    }
    return scaled_copy;
}

int* argmax_row(double **matrix, int n_samples, int n_classes) {
    /* Given a matrix, returns an array with the largest value in each row.*/

    int* argmaxes = malloc(sizeof(int) * n_samples);
    for (int i = 0; i < n_samples; i++) {
        int argmax = 0;
        double max = matrix[i][0];

        for (int j = 1; j < n_classes; j++) {
            if (matrix[i][j] > max) {
                max = matrix[i][j];
                argmax = j;
            }
        }
        argmaxes[i] = argmax;
    }
    return argmaxes;
}

double accuracy(double **predictions, int *labels, int n_samples, int n_classes) {
    /* Compares the argmax from each row in **predictions with the values in
    *labels.*/

    int* y_hat = argmax_row(predictions, n_samples, n_classes);
    int hits = 0;
    for (int i = 0; i < n_samples; i++) {
        hits += (labels[i] == y_hat[i]);
    }

    free(y_hat);
    return (double)hits / n_samples;
}

#endif
