#!/bin/bash
for dataset in dblp acm imdb cora citeseer pubmed; do
for gs in 8 16 32 64 128; do
    echo "Running on dataset: $dataset gs=$gs"
    python attack.py --dataset $dataset --group_size $gs >> "${dataset}-attack.log"
done
done





