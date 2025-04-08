# !/bin/sh

bptts=(1)
nodes=(15 30 100 200)

for bptt in "${bptts[@]}"; do
    for node in "${nodes[@]}"; do
        echo "Running script for bptt $bptt node $node"
        python generate_cluster_result_CL.py --bptt $bptt --node $node
    done
done