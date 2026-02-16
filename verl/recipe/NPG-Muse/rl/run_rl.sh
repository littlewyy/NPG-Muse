
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export SAVE_FREQ=${SAVE_FREQ:-25} 
export RAY_DISABLE_MEMORY_MONITOR=1
export HYDRA_FULL_ERROR=1
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}


BASE_MODEL_PATH=/path/to/your/sft_model
REWARD_TYPE=complicated 
DATA_VOLUME=45000 


if [ ! -e "$BASE_MODEL_PATH" ]; then
    echo "Error: BASE_MODEL_PATH not found: $BASE_MODEL_PATH"
    exit 1
fi

MODEL_NAME=$(basename "$BASE_MODEL_PATH")

PROJECT_NAME='npg-muse-rl' 
EXPERIMENT_NAME="RL_${REWARD_TYPE}_${MODEL_NAME}_${DATA_VOLUME}" 
CHECKPOINT_DIR=./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
MAX_PROMPT_LENGTH=5000 

DATA_BASE_PATH=./npg_muse_attachments/training_data/RL_data/reward_${REWARD_TYPE}
TRAIN_DATA="${DATA_BASE_PATH}/train_graph_3np_all_levels_${DATA_VOLUME}.parquet"
TEST_DATA="${DATA_BASE_PATH}/valid_graph.parquet"
TEMPERATURES=(1.0)
RESUME_MODE="disable"  
MAX_RESPONSE_LENGTHS=(8192)  
SP=(4)        
VAL_TEMP=0.8
OFFLOAD_REF=False 
test_freq=$SAVE_FREQ
save_freq=$SAVE_FREQ
BATCH_SIZE=128


if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data file not found: $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "Error: Test data file not found: $TEST_DATA"
    exit 1
fi


BASE_CONFIG="\
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${BATCH_SIZE} \
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
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    "
mkdir -p ${CHECKPOINT_DIR}

echo "======= STARTING MIXED TRAINING ======="
echo "Max response lengths: ${MAX_RESPONSE_LENGTHS[@]}"
echo "Temperatures: ${TEMPERATURES[@]}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"

max_response_length=${MAX_RESPONSE_LENGTHS[0]}
temperature=${TEMPERATURES[0]}
sp_size=${SP[0]}

COMMAND="python3 -m verl.trainer.main_ppo \
${BASE_CONFIG} \
data.val_files=['$TEST_DATA'] \
data.train_files=['$TRAIN_DATA'] \
data.max_response_length=${max_response_length} \
actor_rollout_ref.rollout.max_num_batched_tokens=$((max_response_length + MAX_PROMPT_LENGTH)) \
actor_rollout_ref.rollout.temperature=${temperature} \
trainer.resume_mode=$RESUME_MODE \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
trainer.total_epochs=1"


$COMMAND 2>&1

echo "======= FULL TRAINING COMPLETED ======="
echo "Final checkpoint saved in: ${CHECKPOINT_DIR}" 