python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /path/to/checkpoint/global_step_xxx \
    --target_dir /path/to/model_dir/model_name