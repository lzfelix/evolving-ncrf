#!/bin/bash

MAX_AMOUNT=15
WEIGHTS_LINE=1

echo "Usage: ./train_validate model_dataset, ie: softmax_ds3"
echo "If optimizing for F1 score, do not forget to activate the venv"
echo

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit -1
fi

for i in `seq 1 $MAX_AMOUNT`;
do
    echo "Training $i/$MAX_AMOUNT"
    filename="output_$i-$1.txt"
    ./bin/optimizer_ga $1 > $filename 2>&1
    
    tac $filename | sed "${WEIGHTS_LINE}q;d"

    rm $filename
    echo 
done
