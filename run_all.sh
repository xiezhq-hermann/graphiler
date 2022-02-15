#!/bin/bash
for model in GAT GCN HGT RGCN
do
  python examples/${model}/${model}.py all 0
  ./visualize ${model}
done
