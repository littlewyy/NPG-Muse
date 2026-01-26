# 用最佳settings在不同series models上训练
BEST_STRATEGY=$1
BEST_REWARD=$2
BEST_DATA_VOLUME=$3
MODEL_PATHS = (
    "npg_muse_attachments/sft_models/qwen25_1m_sft_6np",
    "npg_muse_attachments/sft_models/llama31_sft_6np",
    "npg_muse_attachments/sft_models/g1_sft_6np",
    "npg_muse_attachments/sft_models/Qwen3-8B-Base",
)
for model_path in $MODEL_PATHS; do
    bash recipe/NPG-Muse/rl/6np_$BEST_STRATEGY.sh $model_path $BEST_REWARD $BEST_DATA_VOLUME
done