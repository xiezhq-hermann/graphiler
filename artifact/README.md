## MLSys'22 Artifact Evaluation

For docker:
```
docker run --gpus all -i -t -v $(pwd)/output:/root/graphiler/output graphiler artifact/run_all.sh
```

For local build:
```
# create directory storing outputs
mkdir -p output

# benchmark all GAT implementation on all datasets
python $GRAPHILER/examples/GAT/GAT.py all 0
./visualize.sh GAT

# run all experiments and visualize results
export REPEAT=50  # manually specify the number of repeats, you can change it to whatever you want.
./run_all.sh
Note: The number of repeats in docker was set to `50` by default.
```

More instructions are on the way : )