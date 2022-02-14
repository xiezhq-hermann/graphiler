#!/bin/bash
for model in GAT GCN HGT RGCN CGAT
do
  python examples/${model}/${model}.py all 0
  python scripts/plot.py --model ${model} --time
  python scripts/plot.py --model ${model} --mem
done
