# Before running, fill in rdzv_id, rdzv_backend, rdzv_endpoint. This should be run across two nodes.
axolotl train $1 \
        --launcher torchrun -- --nnodes 2 --nproc_per_node 8 --rdzv_id "" --rdzv_backend "" --rdzv_endpoint ""