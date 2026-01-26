DATA_VOLUME=$1
BASE_MODEL_PATH=npg_muse_attachments/sft_models/qwen3_sft_6np
bash recipe/NPG-Muse/rl/6np_mixed.sh ${BASE_MODEL_PATH} binary ${DATA_VOLUME}