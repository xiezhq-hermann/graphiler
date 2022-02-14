for model in GAT GCN HGT RGCN CGAT
do
  python scripts/plot.py --model ${model} --time
  python scripts/plot.py --model ${model} --mem
done