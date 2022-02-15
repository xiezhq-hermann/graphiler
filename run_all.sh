#!/bin/bash
for model in GAT GCN HGT RGCN CGAT
do
  python examples/${model}/${model}.py all 0
  ./visualize ${model}
done
