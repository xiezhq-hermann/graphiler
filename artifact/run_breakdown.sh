#!/bin/bash
for model in GAT HGT
do
    python $GRAPHILER/examples/${model}/${model}.py breakdown 0
    bash $GRAPHILER/artifact/visualize.sh ${model}_breakdown
done
