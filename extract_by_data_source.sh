#!/bin/bash

# 使用示例:
# bash extract_by_data_source.sh input.parquet output.parquet source1 source2 ...

# if [ "$#" -lt 3 ]; then
#     echo "用法: $0 <输入文件> <输出文件> <数据源1> [数据源2 ...]"
#     exit 1
# fi

SRC=verl/npg_muse_attachments/training_data/RL_data/reward_binary/train_graph_6np_all_levels_750.parquet
DST=verl/npg_muse_attachments/training_data/RL_data/reward_binary/train_graph_3np_all_levels_150.parquet
SOURCES=("TSP_binary" "MCP_binary" "GED_binary")

python3 extract_by_data_source.py --src "$SRC" --dst "$DST" --sources "${SOURCES[@]}"
