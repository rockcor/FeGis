#!/bin/bash
for dataset in dblp cora citeseer pubmed; do
    echo "Running on dataset: $dataset"
    python attack.py --dataset $dataset --num_view 3 --attack 0>> "${dataset}_nv_3.log"
done





