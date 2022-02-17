#!/bin/bash
for model in GAT GCN HGT RGCN
do
    python $GRAPHILER/examples/${model}/${model}.py all 0
    bash $GRAPHILER/artifact/visualize.sh ${model}
done
