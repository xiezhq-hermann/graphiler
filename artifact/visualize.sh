#!/bin/bash
python $GRAPHILER/artifact/plot.py --model $1 --time
python $GRAPHILER/artifact/plot.py --model $1 --mem
