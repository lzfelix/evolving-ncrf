// Python includes should be the first one
#include "f1.h"

// LibOPT includes
#include "common.h"
#include "function.h"
#include "gp.h"
#include "pso.h"

#include <stdio.h>
#include <float.h>
#include <stdbool.h>

// Custom includes
#include "matrix.h"
#include "file_utils.h"

// GP hyperparameters
#define N_PARTICLES    20
#define N_ITERATIONS   1000 //1000
#define P_MUTATION     0.1
#define P_CROSSOVER    NAN // these values do not matter for GP
#define P_REPRODUCTION NAN

// Number of candidates models to be used
#define NC 12

// Do not forget to update these macros!
#define USE_F1 false
#define USE_LAMBDA 0


double convex_combination(Agent *a, va_list extra_args) {
    // Getting extra parameters from varargs. The a->n == #candidates.
    // call: runGA(s, convex_combination, buffer, all_predictions, labels, n_samples, n_classes);
    double **buffer = va_arg(extra_args, double**);
    double ***all_predictions = va_arg(extra_args, double***);
    int *ground_truth = va_arg(extra_args, int*);
    int n_samples = va_arg(extra_args, int);
    int n_classes = va_arg(extra_args, int);

    double lambda = -1;
    if (USE_LAMBDA) {
        lambda = a->x[a->n - 1];
    }
    // Now, here's the drill:
    //  all_predictions is an array of matrices (one-hot encoded predictions). Since C99
    //      does not allow dinamically sized array declarations, we can fallback to pointers,
    //      since, under the hood, arrays are pointers. Hence double ***x is the same as
    //      double **x[a->n], and **x is just a matrix that can be accessed as x[i][j].

    // A += (d * P)
    // y_hat[i] = argmax(A)[i]

    // Normalizing weights to add up to 1
    double mass = 0;
    for (int i = 0; i < a->n - USE_LAMBDA; i++) {
        mass += a->x[i];
    }

    for (int i = 0; i < a->n - USE_LAMBDA; i++) {
        if (a->x[i] >= lambda) {
            buffer = mult_add_store(a->x[i] / mass, all_predictions[i], buffer, n_samples, n_classes);
        }
    }

    double fitness;
    if (USE_F1) {
        int* y_hat = argmax_row(buffer, n_samples, n_classes);
        fitness = -1 * f1_score(ground_truth, y_hat, n_samples, "macro");
        free(y_hat);
    }
    else {
        fitness = -1 * accuracy(buffer, ground_truth, n_samples, n_classes);
    }

    // Empty the buffer before evaluating the next agent
    buffer = restore(buffer, n_samples, n_classes);
    return fitness;
}

int main(int argc, const char* argv[]) {
    int n_samples, n_classes;
    int *labels = NULL;
 
    if (argc != 2) {
        fprintf(stderr, "Usage: optimizer dataset_name\n");
        exit(-1);
    }

    if (USE_F1) {
        fprintf(stdout, "This program minimizes macro-F1\n");
    }

    int is_one_hot = 0;
    char labels_filepath[90];
    char *filepaths[NC];
    
    for (int i = 0; i < NC; i++) {
        filepaths[i] = malloc(sizeof(char) * 200);
    }

    if (strcmp(argv[1], "dsC_softmax") == 0) {
        fprintf(stdout, "[!] Using softmax_ds1\n");
        is_one_hot = 0;

        strcpy(labels_filepath, "../find_candidates/cand_predictions/dsC_softmax/labels.txt");
        
        strcpy(filepaths[0], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 20:00:27.950319_epoch=14.txt");
		strcpy(filepaths[1], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 20:00:27.950319_epoch=22.txt");
		strcpy(filepaths[2], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 20:00:27.950319_epoch=2.txt");
		strcpy(filepaths[3], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 20:00:27.950319_epoch=6.txt");
		strcpy(filepaths[4], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 20:45:19.299095_epoch=13.txt");
		strcpy(filepaths[5], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 20:45:19.299095_epoch=21.txt");
		strcpy(filepaths[6], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 20:45:19.299095_epoch=2.txt");
		strcpy(filepaths[7], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 20:45:19.299095_epoch=6.txt");
		strcpy(filepaths[8], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 21:38:39.965463_epoch=12.txt");
		strcpy(filepaths[9], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 21:38:39.965463_epoch=2.txt");
		strcpy(filepaths[10], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 21:38:39.965463_epoch=30.txt");
		strcpy(filepaths[11], "../find_candidates/cand_predictions/dsC_softmax/partial_2019-03-24 21:38:39.965463_epoch=5.txt");
    }
    else if (strcmp(argv[1], "dsC_crf") == 0) {
        fprintf(stdout, "[!] Using crf_ds1\n");
        is_one_hot = 1;

        strcpy(labels_filepath, "../find_candidates/cand_predictions/dsC_crf/labels.txt");

        strcpy(filepaths[0], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 15:06:11.592878_epoch=14.txt");
        strcpy(filepaths[1], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 15:06:11.592878_epoch=28.txt");
        strcpy(filepaths[2], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 15:06:11.592878_epoch=2.txt");
        strcpy(filepaths[3], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 15:06:11.592878_epoch=6.txt");

        strcpy(filepaths[4], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 15:22:24.154152_epoch=11.txt");
        strcpy(filepaths[5], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 15:22:24.154152_epoch=24.txt");
        strcpy(filepaths[6], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 15:22:24.154152_epoch=2.txt");
        strcpy(filepaths[7], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 15:22:24.154152_epoch=6.txt");

        strcpy(filepaths[8],  "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 16:27:09.800280_epoch=14.txt");
        strcpy(filepaths[9],  "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 16:27:09.800280_epoch=2.txt");
        strcpy(filepaths[10], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 16:27:09.800280_epoch=30.txt");
        strcpy(filepaths[11], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-25 16:27:09.800280_epoch=6.txt");
    }
    else if (strcmp(argv[1], "dsD_softmax") == 0) {
        fprintf(stdout, "[!] Using softmax_ds3\n");
        is_one_hot = 0;

        strcpy(labels_filepath, "../find_candidates/cand_predictions/dsD_softmax/labels.txt");

        strcpy(filepaths[0], "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 13:39:44.599061_epoch=12.txt");
        strcpy(filepaths[1], "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 13:39:44.599061_epoch=24.txt");
        strcpy(filepaths[2], "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 13:39:44.599061_epoch=2.txt");
        strcpy(filepaths[3], "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 13:39:44.599061_epoch=3.txt");

        strcpy(filepaths[4], "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 14:17:42.703867_epoch=14.txt");
        strcpy(filepaths[5], "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 14:17:42.703867_epoch=17.txt");
        strcpy(filepaths[6], "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 14:17:42.703867_epoch=2.txt");
        strcpy(filepaths[7], "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 14:17:42.703867_epoch=6.txt");

        strcpy(filepaths[8],  "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 14:38:24.681469_epoch=10.txt");
        strcpy(filepaths[9],  "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 14:38:24.681469_epoch=26.txt");
        strcpy(filepaths[10], "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 14:38:24.681469_epoch=2.txt");
        strcpy(filepaths[11], "../find_candidates/cand_predictions/dsD_softmax/partial_2019-03-23 14:38:24.681469_epoch=6.txt");
    }
    else if (strcmp(argv[1], "dsD_crf") == 0) {
        fprintf(stdout, "[!] Using crf_ds3\n");
        is_one_hot = 1;

        strcpy(labels_filepath, "../find_candidates/cand_predictions/dsD_crf/labels.txt");

        strcpy(filepaths[0], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 13:06:27.737145_epoch=26.txt");
        strcpy(filepaths[1], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 13:06:27.737145_epoch=2.txt");
        strcpy(filepaths[2], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 13:06:27.737145_epoch=6.txt");
        strcpy(filepaths[3], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 13:06:27.737145_epoch=9.txt");

        strcpy(filepaths[4], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 14:01:38.313193_epoch=12.txt");
        strcpy(filepaths[5], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 14:01:38.313193_epoch=23.txt");
        strcpy(filepaths[6], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 14:01:38.313193_epoch=2.txt");
        strcpy(filepaths[7], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 14:01:38.313193_epoch=5.txt");

        strcpy(filepaths[8],  "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 14:22:04.061383_epoch=11.txt");
        strcpy(filepaths[9],  "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 14:22:04.061383_epoch=26.txt");
        strcpy(filepaths[10], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 14:22:04.061383_epoch=2.txt");
        strcpy(filepaths[11], "../find_candidates/cand_predictions/dsD_crf/partial_2019-03-18 14:22:04.061383_epoch=6.txt");
    }
    else {
        fprintf(stderr, "Invalid dataset name. Valid values are: [dsC|dsD]_[softmax|crf]\n");
        exit(-1);
    }

    int n_candidates = NC;
    double ***all_predictions = load_multiple_predictions(filepaths,
                                                          n_candidates,
                                                          &n_samples,
                                                          &n_classes,
                                                          is_one_hot);
    n_candidates += USE_LAMBDA;

    // Loading ground truth labels
    labels = load_labels(labels_filepath, n_samples, n_classes);

    //---------------------------------------------------------------------------------
    // GP stuff
    int min_depth = 1, max_depth = 3, n_terminals = 2, i, j;
    int n_functions = 3;
    char **terminal = NULL, **function = NULL;
    double **constant = NULL;

    /* Built-in features ******************************************************/
    /* loading set of terminals */
    terminal = (char **)malloc(n_terminals * sizeof(char *));
    for (i = 0; i < n_terminals; i++)
        terminal[i] = (char *)malloc(TERMINAL_LENGTH * sizeof(char));
    strcpy(terminal[0], "(x,y)");
    strcpy(terminal[1], "CONST");
    /****************************/

    /* loading set of functions */
    function = (char **)malloc(n_functions * sizeof(char *));
    for (i = 0; i < n_functions; i++)
        function[i] = (char *)malloc(TERMINAL_LENGTH * sizeof(char));
    strcpy(function[0], "SUM");
    strcpy(function[1], "SUB");
    strcpy(function[2], "MUL");
    /****************************/

    /* loading constants */
    constant = (double **)malloc(n_candidates * sizeof(double *));
    for (i = 0; i < n_candidates; i++)
    {
        constant[i] = (double *)malloc(N_CONSTANTS * sizeof(double));
        for (j = 0; j < N_CONSTANTS; j++)
            constant[i][j] = GenerateUniformRandomNumber(-5.0, 5.0);
    }
    /*********************/

    // GP - Creating the seach space
    SearchSpace *s = CreateSearchSpace(N_PARTICLES, n_candidates, _GP_, min_depth, max_depth, n_terminals, N_CONSTANTS, n_functions, terminal, constant, function);
    s->iterations = N_ITERATIONS;// [N_ITERATIONS, 10]
    s->is_integer_opt = 0; /* integer(binary)-valued optimization problem */


    // GP - Initializing search space. All models importance is between [0, 1]
    for (int i = 0; i < n_candidates; i++) {
        s->LB[i] = 0;
        s->UB[i] = 1;
    }

    s->pReproduction = 0.3;                                /* Setting up the probability of reproduction */
    s->pMutation = 0.4;                                    /* Setting up the probability of mutation */
    s->pCrossover = 1 - (s->pReproduction + s->pMutation); /* Setting up the probability of crossover */

    // GP - Checking if everything is okay
    InitializeSearchSpace(s, _GP_);
    if (!CheckSearchSpace(s, _GP_)) {
        fprintf(stderr, "Invalid search space configuration.\n");
        DestroySearchSpace(&s, _GP_);
        exit(-1);
    }

    //---------------------------------------------------------------------------------

    // GP - Evolve!
    double **buffer = allocate_matrix(n_samples, n_classes);
    if (USE_F1) {
        start_python();
    }
    runGP(s, convex_combination, buffer, all_predictions, labels, n_samples, n_classes);
    if (USE_F1) {
        end_python();
    }

    // GP - Show results
    fprintf(stdout, "\nCandidates importance:\n");
    for (int i = 0; i < n_candidates - USE_LAMBDA; i++) {
        fprintf(stdout, "\t - %f x %s\n", s->g[i], filepaths[i]);
    }
    if (USE_LAMBDA) {
        fprintf(stdout, "\t - Lambda = %f\n", s->g[n_candidates - 1]); 
    }

    fprintf(stdout, "w = [");
    for (int i = 0; i < n_candidates - USE_LAMBDA; i++) {
        fprintf(stdout, "%lf, ", s->g[i]);
    }
    fprintf(stdout, "]\n");
    if (USE_LAMBDA) {
        fprintf(stdout, "cutoff = %lf\n\n", s->g[n_candidates- 1]);
    }

    n_candidates -= USE_LAMBDA;

    // Clean up
    DestroySearchSpace(&s, _GP_);
    free(labels);
    destroy_matrix(buffer, n_samples);
    destroy_matrix_list(all_predictions, n_candidates, n_samples);

    // Freeing up filepaths array
    for (int i = 0; i < NC; i++) {
        free(filepaths[i]);
    }

    return 0;
}
