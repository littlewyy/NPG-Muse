#!/bin/bash
set -x
# export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_DISABLE_MEMORY_MONITOR=1
export HYDRA_FULL_ERROR=1

BASE_MODEL_PATH=$1 # 从参数中读取模型路径
REWARD_TYPE=$2 # 从参数中读取reward类型
DATA_VOLUME=$3 # 从参数中读取数据量
require_arg() {
    local name=$1
    local value=$2
    if [ -z "$value" ]; then
        echo "Error: Missing required argument: $name"
        echo "Usage: $0 <BASE_MODEL_PATH> <REWARD_TYPE> <DATA_VOLUME>"
        exit 1  
    fi
}
require_arg "BASE_MODEL_PATH" "$BASE_MODEL_PATH"
require_arg "REWARD_TYPE" "$REWARD_TYPE"
require_arg "DATA_VOLUME" "$DATA_VOLUME"

if [ ! -e "$BASE_MODEL_PATH" ]; then
    echo "Error: BASE_MODEL_PATH not found: $BASE_MODEL_PATH"
    exit 1
fi

MODEL_NAME=$(basename "$BASE_MODEL_PATH")
DATA_PER_LEVEL=$((DATA_VOLUME/5))

# Training parameters
PROJECT_NAME='npg-muse-rl' 
EXPERIMENT_NAME="6np_curriculum_${REWARD_TYPE}_${MODEL_NAME}_${DATA_VOLUME}" 
CHECKPOINT_DIR=./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
MAX_PROMPT_LENGTH=5000 # 试试看能否容纳所有的数据，同时能跑
VAL_TEMP=0.8
OFFLOAD_REF=False # 如果显存不足，再改成True
test_freq=25
save_freq=-1
BATCH_SIZE=128

RESUME_MODE="disable"  # Initial mode for first level
TOTAL_EPOCHS=1      # 初始化总epoch数
LEVELS=(1 2 3 4 5)  # 定义训练级别
MAX_RESPONSE_LENGTHS=(8192 8192 8192 8192 8192)  # Increasing response lengths
SP=(4 4 4 4 4)        # Sequence parallel size (can be adjusted per level)
TEMPERATURES=(1.0 1.0 1.0 1.0 1.0)  

# Data path configuration - please modify according to your actual setup
DATA_BASE_PATH=./npg_muse_attachments/training_data/RL_data/reward_${REWARD_TYPE}

# Complete BASE_CONFIG (preserve all original parameters)
BASE_CONFIG="\
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMP \
    actor_rollout_ref.model.path=$BASE_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD_REF \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    "

# Create checkpoint directory if it doesn't exist
mkdir -p ${CHECKPOINT_DIR}

echo "======= STARTING FULL CURRICULUM LEARNING ======="
echo "Total levels to train: ${#LEVELS[@]}"
echo "Levels: ${LEVELS[@]}"
echo "Max response lengths: ${MAX_RESPONSE_LENGTHS[@]}"
echo "Temperatures: ${TEMPERATURES[@]}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"

for i in "${!LEVELS[@]}"; do
    level=${LEVELS[$i]}
    max_response_length=${MAX_RESPONSE_LENGTHS[$i]}
    temperature=${TEMPERATURES[$i]}
    sp_size=${SP[$i]}
    
    echo "======= STARTING LEVEL $level (${i+1}/${#LEVELS[@]}) ======="
    echo "Level: $level"
    echo "Max response length: $max_response_length"
    echo "Temperature: $temperature"
    echo "Sequence parallel size: $sp_size"
    echo "Resume mode: $RESUME_MODE"
    
    # Data file paths - please modify according to your actual setup
    TRAIN_DATA="${DATA_BASE_PATH}/train_graph_6np_level_${level}_${DATA_PER_LEVEL}.parquet"
    TEST_DATA="${DATA_BASE_PATH}/valid_graph.parquet"

    # Parameter validation
    if [ ! -f "$TRAIN_DATA" ]; then
        echo "Error: Training data file not found: $TRAIN_DATA"
        exit 1
    fi

    if [ ! -f "$TEST_DATA" ]; then
        echo "Error: Test data file not found: $TEST_DATA"
        exit 1
    fi

    COMMAND="python3 -m verl.trainer.main_ppo \
        ${BASE_CONFIG} \
        data.train_files=['$TRAIN_DATA'] \
        data.val_files=['$TEST_DATA'] \
        data.max_response_length=${max_response_length} \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_response_length + MAX_PROMPT_LENGTH)) \
        actor_rollout_ref.rollout.temperature=${temperature} \
        trainer.resume_mode=$RESUME_MODE \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
        trainer.total_epochs=$TOTAL_EPOCHS"
            
    # Execute training command
    $COMMAND 2>&1
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "======= LEVEL $level COMPLETED SUCCESSFULLY ======="
    else
        echo "======= LEVEL $level FAILED ======="
        echo "Training failed for level $level. Check the log file for details."
        exit 1
    fi

    # Update resume mode for next level (auto resume from second level)
    RESUME_MODE="auto"
    
    # Optional: Clear GPU cache between levels
    # nvidia-smi --gpu-reset
    
    # Increment epochs for next level
    TOTAL_EPOCHS=$((TOTAL_EPOCHS + 1))
    
    echo "======= LEVEL $level FINISHED ======="
    echo ""
done

echo "======= FULL CURRICULUM LEARNING COMPLETED ======="
echo "All levels have been trained successfully!"
echo "Final checkpoint saved in: ${CHECKPOINT_DIR}" 

STEP_PER_LEVEL=$((DATA_PER_LEVEL/${BATCH_SIZE}))
NUMBER_STEPS=$((STEP_PER_LEVEL*5))
CHECKPOINT_PATH="${CHECKPOINT_DIR}/global_step_${NUMBER_STEPS}"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_PATH"
    exit 1
fi
# 顺便把checkpoint转成模型权重
python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir "$CHECKPOINT_PATH" \
    --target_dir ./npg_muse_attachments/rl_models/${EXPERIMENT_NAME}