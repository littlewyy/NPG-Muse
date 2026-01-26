# for i in 1 2 3 4 5; do
#     python extract_parquet_ratio.py \
#         --src_path training_data/RL_data/train_graph_6np_level_${i}_18000_cleaned.parquet \
#         --dst_path training_data/RL_data/train_graph_6np_level_${i}_6000.parquet \
#         --ratio 0.3333333333333333
# done

# for i in 1 2 3 4 5; do
#     python extract_parquet_ratio.py \
#         --src_path verl/npg_muse_attachments/training_data/RL_data/train_graph_6np_level_${i}_18000.parquet \
#         --dst_path verl/npg_muse_attachments/training_data/RL_data/train_graph_6np_level_${i}_9000.parquet \
#         --ratio 0.5
# done

# python merge_parquet.py \
#     --input verl/npg_muse_attachments/training_data/RL_data/train_graph_6np_level_*_9000.parquet \
#     --output verl/npg_muse_attachments/training_data/RL_data/train_graph_6np_all_levels_45000.parquet

for i in 1 2 3 4 5; do
    python extract_parquet_ratio.py \
        --src_path verl/npg_muse_attachments/training_data/RL_data/train_graph_6np_level_${i}_18000.parquet \
        --dst_path verl/npg_muse_attachments/training_data/RL_data/train_graph_6np_level_${i}_150.parquet \
        --ratio 0.00833333333333333
done

python merge_parquet.py \
    --input verl/npg_muse_attachments/training_data/RL_data/train_graph_6np_level_*_150.parquet \
    --output verl/npg_muse_attachments/training_data/RL_data/train_graph_6np_all_levels_750.parquet