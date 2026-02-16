<div align="center">

# NPG-Muse: Scaling Long Chain-of-Thought Reasoning with NP-Hard Graph Problems

</div>

## Models

| Model                                                             | Base                   | Parameters |
| ----------------------------------------------------------------- | ---------------------- | ---------- |
| [NPG-Muse-7B](http://modelscope.cn/models/littlewyy/NPG-Muse-7B)     | Qwen2.5-7B-Instruct-1M | 7.62B      |
| [NPG-Muse-8B](http://modelscope.cn/models/littlewyy/NPG-Muse-8B) | Qwen3-8B           | 8.2B      |

## Quick Start on Training


### Training Environment Setup

```bash
conda create -n npg-muse python=3.10
conda activate npg-muse
cd ./verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pip install string-repetition==0.1.0 rdkit walker munkres fast_tsp
pip install modelscope[framework]
```


### Download Source Data

Before starting training, you need to download and prepare the data files. These files are large and have been excluded from the repository to keep it lightweight.

```bash
cd verl
mkdir -p npg_muse_attachments
cd npg_muse_attachments
mkdir -p training_data
cd training_data
mkdir -p SFT_data
cd SFT_data
modelscope download --dataset 'littlewyy/NPG-Muse-SFT-data' sft_data_npg_3np.parquet --local_dir './'
modelscope download --dataset 'littlewyy/NPG-Muse-SFT-data' sft_data_npg_val.parquet --local_dir './'
cd ../
mkdir -p RL_data
cd RL_data
mkdir -p reward_complicated
cd reward_complicated
modelscope download --dataset 'littlewyy/NPG-Muse-RL-data' train_graph_3np_all_levels_45000.parquet --local_dir './'
modelscope download --dataset 'littlewyy/NPG-Muse-RL-data' valid_graph.parquet --local_dir './'
```

### Two-Stage Training Pipeline

#### Stage 1: Supervised Fine-Tuning

```bash
cd verl
# before running the following bash, modify paths of datasets and base models in sft_trainer.yaml and run_sft.sh
bash verl/recipe/NPG-Muse/sft/run_sft.sh
```

#### Stage 2: Reinforcement Learning with Curriculum Learning

```bash
cd verl
# before running the following bash, replace the BASE_MODEL_PATH with the path of NPG-Muse-Stage1 model in run_rl.sh
bash recipe/NPG-Muse/rl/run_rl.sh
```

## Repository Structure

```
NPG-Muse/
├── README.md
├── requirements.txt
├── verl/
│   └── recipe/
│       └── NPG-Muse/
│           ├── sft/                         # Stage 1: SFT Training
│           │   ├── run_sft.sh               #   Training script
│           │   ├── sft_trainer.yaml         #   Configuration
|           ├──rl/                           # Stage 2: RL Training
│           │   ├── run_rl.sh                #   Training script
```

## Hardware Requirements

### Recommended Configuration

| Component | Specification |
| --------- | ------------- |
| **GPU**   | 8x A800 80GB  |


