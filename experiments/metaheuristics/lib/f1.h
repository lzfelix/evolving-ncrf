// WARNING: When using this library, it *must* be included
// before any other in your program, as Python.h sets up some
// variables that may interfere with other header files

#ifndef __F1_PYTHON__
#define __F1_PYTHON__

#include "Python.h"

void start_python(void) {
    Py_Initialize();
}

void end_python(void) {
    int status = Py_FinalizeEx();
    if (status != 0) {
        fprintf(stderr, "[ERROR] Failed to close the Python interpreter.\n");
        exit(-1);
    }
}

PyObject *to_pList(int predictions[], int n_items) {
    /* Converts a C array into a Python list of longs. If a failure
    happens, the memory is cleaned and this function returns NULL.*/

    PyObject *pList = PyList_New(n_items);
    int status;

    for (int i = 0; i < n_items; i++) {
        // The outter function steal the reference from the inner,
        // there's no need to DECREF.
        status = PyList_SetItem(pList, i, PyLong_FromLong(predictions[i]));

        if (status != 0) {
            fprintf(stderr, "[ERROR] Failed to insert item %d into list\n", i);
            Py_DECREF(pList);
            return NULL;
        }
    }
    return pList;
}

double f1_score(int actual[], int predictions[], int n_items, char* average) {
    /* Returns sklearn.metrics.f1_score(actual, predictions, average=average).
    
    Before invoking this function, ensure that the python interpreter has been
    activated. This function can terminate the program with an error code if
    something goes wrong.
    */
    double macro_f1 = -1;

    // Loading sklearn.metrics module
    PyObject *pModule_name = PyUnicode_DecodeFSDefault("sklearn.metrics");
    PyObject *pModule = PyImport_Import(pModule_name);
    Py_DECREF(pModule_name);

    if (!pModule) {
        fprintf(stderr, "[ERROR] The module sklearn is not installed. Is the venv active?\n");
        Py_XDECREF(pModule);
        exit(-1);
    }
    
    // Getting the F1 function
    PyObject *pF1score = PyObject_GetAttrString(pModule, "f1_score");
    if (!pF1score) {
        fprintf(stderr, "sklern.metrics.f1_score function is not reachable\n");
        Py_DECREF(pModule);
        Py_XDECREF(pF1score);
        exit(-1);
    }
    
    // Building the arguments
    // sklearn.metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', ...)
    PyObject *pFunArgs = PyTuple_New(5);

    PyObject *pReal = to_pList(actual, n_items);
    PyObject *pPred = to_pList(predictions, n_items);
    PyObject *pResult = NULL;
    
    if (!pReal || !pPred) {
        fprintf(stderr, "Failed to convert C arrays into Python lists\n");
    }
    else {
        // References are stolen by SetItem, taking shortcut
        PyTuple_SetItem(pFunArgs, 0, pReal);
        PyTuple_SetItem(pFunArgs, 1, pPred);
        PyTuple_SetItem(pFunArgs, 2, Py_BuildValue(""));
        PyTuple_SetItem(pFunArgs, 3, PyLong_FromLong(1L));
        PyTuple_SetItem(pFunArgs, 4, PyUnicode_FromString(average));

        // Calling F1-score
        pResult = PyObject_CallObject(pF1score, pFunArgs);
    }

    // Freeing resources
    Py_DECREF(pFunArgs);
    Py_DECREF(pF1score);
    Py_DECREF(pModule);

    if (!pResult || PyErr_Occurred()) {
        fprintf(stderr, "Failed to call F1 score\n");
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
        exit(-1);
    }
    
    macro_f1 = PyFloat_AsDouble(pResult);
    Py_DECREF(pResult);
    return macro_f1;
}

#endif