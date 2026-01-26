# 先基于qwen3消融出最佳的实验settings
model_path="npg_muse_attachments/sft_models/qwen3_sft_6np"
for data_volume in 90000 60000 45000; do
    for training_type in mixed curriculum; do
        for reward_type in binary binary_format binary_format_repeat ratio_quality_format_repeat complicated; do
            bash recipe/NPG-Muse/rl/6np_$training_type.sh $model_path $reward_type $data_volume
        done
    done
done