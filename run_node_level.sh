#!/bin/bash
for dataset in dblp acm imdb cora citeseer pubmed; do
    echo "Running on dataset: $dataset "
    python attack.py --dataset $dataset --node_level 1 >> "${dataset}-nodelevel.log"
done





