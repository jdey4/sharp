# !/bin/sh

bptts=(1)
nodes=(5 10 15 20 25 30 35)

for bptt in "${bptts[@]}"; do
    for node in "${nodes[@]}"; do
        echo "Running script for bptt $bptt node $node"
        python generate_colla_result_sleep.py --bptt $bptt --node $node
    done
done