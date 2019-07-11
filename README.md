# Evolving Neural Conditional Random Fields for Drilling Report Classification

This repository contains all the necessary code to run the same experiments described in the paper *"Evolving Neural Conditional Random Fields for Drilling Report Classification"*, however the user must provide the data in the expected format.


## References

If you use our work, please cite us as: ...



## Folders Strucuture

- `data/`
  - `dsC/`  Folder containing the ds-C dataset. Not made publicly available;
  - `datasets.json` Contains specifications for each folder in `data/`
- `embeddings/` Stores pre-trained word embeddings
- `experiments/`
  - `nn/` Contains the neural networks and word embeddings parts of the experiments;
    - `training_logs/` The output generated during model training. Used to find the best candidates to form the ensemble;
  - `find_candidates/` Contains scripts to identify potential candidates to train the ensemble of models;
  - `metaheuristics/` Contains C code used to learn the ensembles.
- `trained_models/` Keras checkpoint results



## Reproducing the results

In order to reproduce the results, the following steps (detailed below) must be completed:

1. Supplying the datat
2. Training the fastText word embeddings;
3. Training the proposed contextual model multiple times to create the genetic ensemble;
4. Finding the Cyclical Learning Rates minima points;
5. Learning and evaluating ensembles with GA and GP



### 1. Supplying the data

The different datasets must be placed under the `data/` folder. Below follows an illustration of this folder:

```
├── datasets.json
├── dsC
│   ├── filename_1.xlsx
│   ├── filename_2.xlsx
│   ├── filename_3.xlsx
```

After providing the data, the file `datasets.json` must be updated to reflect each dataset characteristics using the following template:

```json
{
    "dataset_name": {
        "mode": "used in our custom code",
        "all_files": "path to the folder containing all files",
        "trn_files": ["./dsC/filename_1.xlsx", "./dsC/filename_2.xlsx"],
        "dev_files": ["./dsC/filename_3.xlsx", "./dsC/filename_4.xlsx"],
        "tst_files": ["./dsC/filename_5.xlsx", "./dsC/filename_6.xlsx"],
        "other_classes": ["List of classes to be ignored. Used in our custom code"],
        "class_map": {
            "maps_a_class": "to_other",
            "used_in_our": "custom_code"
        }
    }
}
```



### 2. Training the word embeddings

Inside the folder `experiments/nn` run `python train_ftt.py [dataset_name]`, where `[dataset_name]` must be one of the keys in the file `./data/datasets.json`. Running this command will train the fastText supervised classifier, generating the paper baseline results. This script also saves the datasets in the fastText format. Further, execute the two commands printed in the last two lines of the console to first train the fastText unsupervised word embeddings. The generated files are automatically saved in `./embeddings/`.



### 3. Training the contextual models

By having the trained the word embeddings, the proposed model can be trained by running `python train_model.py` inside the same folder with the different configurations presented in our paper. To see a complete list of possibilities, run this script with the `-h` flag. Overall, the following variants can be trained out of the box (the names in parenthesis are the ones used in the manuscript):

- Without contextual information (Non-contextual Softmax): requires no additional arguments;
- With contextual information (Contextual Softmax): requires the flag `--use_context`;
- With contextual information and CRF (Contextual CRF): requires the flags `--use_context --use_crf`;

If you plan to train the ensemble of classifiers later on, then ensure to save the console output in a text file, for instance, with: 

```
python train_model.py ds3 ../../embeddings/ds3_model.gensim ../../trained_models/ --use_crf --use_context > ./training_logs/softmax_partials_dsC.txt 2>&1 &
```



### 4. Finding the Cyclical Learning Rates minima points 

The next steps uses the scripts in `experiments/find_candidates/`, which find the points where the model achieved the lowest validation loss during training for each learning rate annealing cycle. This can be achieved with  `python find_val_minima.py [training_logs.txt]`, where the log file is the one generated in the previous step (`softmax_partials_dsC.txt`, in this case).

Running this script will display in the stdout a list of model names and the epochs in which it has achieved the lowest validation loss during each learning rate annealing cycle. A few of these lines (i.e. candidates) must be copied (in our experiments we used three of them) and pasted in the `experiments/specs.json` file using the template below (the remaining fields in this file are going to be filled in the following steps):

```json
{
    "dsC_softmax": {
        "ga_weights": {
			"ac_weights_file": null,
			"f1_weights_file": null
        },
        "gp_weights": {
			"ac_weights_file": null,
			"f1_weights_file": null
        },
        "dev": null,
        "tst": null,
        "details": { },
        "models_folder": "path to where the Keras partial models were saved. By default ../trained_models.json",
        "embeddings": "path to the embeddings. For instance ../embeddings/dsC_model.gensim",
        "checkpoints": {
            "candidate_model_filename": [2, 5, 12, "// epochs found previuosly"],
            "partial_2019-03-24 21:38:39.96452_epoch={}.h5": [1, 6, 10, 30],
        }
    }
}
```



Next, each selected epoch for each candidate must be loaded and used to predict in the validation and test sets. The validation predictions are going to be used to learn the ensemble, while the test set predictions are going to be used in the last step to evaluate the learnt ensembles.

To perform this step using the previuos example, use `python predict_on_minima.py dsC_softmax (model_name) dsC (dataset_name) cand_predictions/dsC_softmax`. This will populate the folder `cand_predictions/dsC_softmax/` with the predictions performed by the models under `checkpoints` in the aforementioned JSON file. Further, these filenames must be used to manually update the `experiments/specs.json`, for instance:

```json
{
    "dsC_softmax": {
        "dev": {
            "labels_file": "./find_candidates/cand_predicitions/dsC_softmax/labels.txt",
			"pred_files": [
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 13:39:44.599061_epoch=12.txt",
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 13:39:44.599061_epoch=24.txt",
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 13:39:44.599061_epoch=2.txt",
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 13:39:44.599061_epoch=3.txt",
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 14:17:42.703867_epoch=14.txt",
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 14:17:42.703867_epoch=17.txt",
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 14:17:42.703867_epoch=2.txt",
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 14:17:42.703867_epoch=6.txt",
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 14:38:24.681469_epoch=10.txt",
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 14:38:24.681469_epoch=26.txt",
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 14:38:24.681469_epoch=2.txt",
				"./find_candidates/cand_predicitions/dsC_softmax/partial_2019-03-23 14:38:24.681469_epoch=6.txt"
			]
        },
		"tst": {
			"labels_file": "./find_candidates/cand_predictions/dsC_softmax/test_labels.txt",
			"pred_files": [
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 13:39:44.599061_epoch=12.txt",
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 13:39:44.599061_epoch=24.txt",
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 13:39:44.599061_epoch=2.txt",
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 13:39:44.599061_epoch=3.txt",
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 14:17:42.703867_epoch=14.txt",
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 14:17:42.703867_epoch=17.txt",
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 14:17:42.703867_epoch=2.txt",
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 14:17:42.703867_epoch=6.txt",
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 14:38:24.681469_epoch=10.txt",
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 14:38:24.681469_epoch=26.txt",
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 14:38:24.681469_epoch=2.txt",
				"./find_candidates/cand_predictions/dsC_softmax/test_partial_2019-03-23 14:38:24.681469_epoch=6.txt"
			]
		},
    }
}
```



### 5. Learning and evaluating ensembles with GA and GP

In order to complete this step the following prerequisites must be fulfilled within the folder `metaheuristics/`:

* Install the [LibOPT](https://github.com/jppbsi/libopt) library:
  * Just clone the repository and run `make`;
  * Update the variable `OPT_HOME` in our `Makefile`;
* Install the python3.6-dev headers:
  * Update the ppa: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt-get update`;
  * Install the headers: `sudo apt-get install python3.6-dev`;
  * Update the variables `PYTHON_INCLUDE` and `PYTHON_PATH` in our `Makefile`;
  * You might want to change the GCC compiler used in the variable `CC` (old gcc compilers may not work);
* **NOTE:** The predictions added in the `dev` entry in the previuos JSON file must be manually copied to the `gp.c` and `ga.c` programs. Remember that changing the contents in the programs requires making the binaries once again.
* It is possible to chose optimizing F1 instead of accuracy by changing the line `#define USE_F1 true` in `src/ga.c` and `/src/gp.c` files;
* Compile the GA and GP optimization programs by running `make`.  
  

The GA and GP ensembles can be learned with the bash scripts in this folder (`train_validate.sh` runs the GA 15 times) and `train_validate_gp.sh` (does the same using Genetic Programming). These scripts print in the screen the weights for each candidate found in each run. Since such information is required to evaluate the embeddings, their output should be redirected to a file as follows:

```bash
./train_validate.sh dsC_softmax > dsC_softmax_ga_acc_weights.txt 2>&1 &
```

The filename `dsC_softmax_ga_acc_weights.txt` should be placed in the `specs.json` file under `dsC_softmax:ac_weights_file` . Finally, run `python apply_weights.py` in order to evaluate the learned ensembles. Using the flag `-h` shows the available options. The baseline results can be obtained with  `python baselines.py`. Likewise, using the flag `-h` shows the different running options.
