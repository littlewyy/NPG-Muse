# RL训练
## 安装环境，并进入verl目录
```bash
conda create -n npg-muse python=3.10
conda activate npg-muse
cd ./verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pip install string-repetition==0.1.0 rdkit walker munkres fast_tsp
```
## 下载基模、数据，运行小规模验证实验
- 首先安装modelscope
```bash
pip install modelscope[framework]
```
- 通过modelscope下载模型和数据集到指定路径
```bash
mkdir -p npg_muse_attachments
cd npg_muse_attachments
mkdir -p sft_models
cd sft_models
modelscope download --model 'littlewyy/qwen3_sft_3np' --local_dir ./qwen3_sft_3np
cd ../
mkdir -p training_data
cd training_data
mkdir -p RL_data
cd RL_data
mkdir -p reward_binary
cd reward_binary
modelscope download --dataset 'littlewyy/3np_360' train_graph_3np_all_levels_360.parquet --local_dir './'
modelscope download --dataset 'littlewyy/3np_45000' train_graph_3np_all_levels_45000.parquet --local_dir './'
modelscope download --dataset 'littlewyy/3np_360' valid_graph.parquet --local_dir './'
```
- 回到仓库根目录，运行以下命令。如果运行完后能在`verl/npg_muse_attachments/rl_models`中找到模型`3np_mixed_binary_qwen3_sft_3np_360`，说明运行成功。
```bash
cd verl
export SAVE_FREQ=1
bash recipe/NPG-Muse/rl/3np_mixed.sh npg_muse_attachments/sft_models/qwen3_sft_3np binary 360
```
- 如果rdkit出现错误`ImportError: libXrender.so.1: cannot open shared object file: No such file or directory`，则:
```bash
# 1. Ubuntu / Debian 
sudo apt-get update
sudo apt-get install -y libxrender1 libxext6 libsm6
# 2. CentOS / RHEL / Alma / Rocky
sudo yum install -y libXrender libXext libSM
# 或者用 dnf
sudo dnf install -y libXrender libXext libSM
# 3. 如果没有sudo权限，则用conda安装
conda install -y -c conda-forge libxrender libxext libsm
```
## 正式训练（预估时间：5-7天）
回到仓库根目录，运行以下命令
```bash
export SAVE_FREQ=50 # 总共有350步，为了以防万一存一下中间的checkpoint。如果没空间可以把此处改为-1，只存最后一个checkpoint
bash recipe/NPG-Muse/rl/3np_mixed.sh npg_muse_attachments/sft_models/qwen3_sft_3np binary 45000
```