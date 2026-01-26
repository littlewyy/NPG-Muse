python merge_parquet.py \
    --input training_data/RL_data/train_graph_6np_level_*_6000.parquet \
    --output training_data/RL_data/train_graph_6np_all_levels_30000.parquet
python merge_parquet.py \
    --input training_data/RL_data/train_graph_6np_level_*_12000.parquet \
    --output training_data/RL_data/train_graph_6np_all_levels_60000.parquet