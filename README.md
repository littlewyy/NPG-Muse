<div align="center">

# NPG-Muse: Scaling Long Chain-of-Thought Reasoning with NP-Hard Graph Problems

</div>



Reasoning Large Language Models (RLLMs) have recently achieved remarkable progress on complex reasoning tasks, largely enabled by their long chain-of-thought (**Long CoT**) capabilities. However, developing these Long CoT behaviors relies heavily on post-training with high-quality datasets, which are typically costly and human-curated (e.g., mathematics and code), leaving **scalable** alternatives unexplored. In this work, we introduce **NP-hard (NPH) graph problems** as a novel synthetic training corpus, as they inherently require deep reasoning, extensive exploration, and reflective strategies—the core characteristics of Long CoT reasoning. Building on this insight, we develop a two-stage post-training framework: (i) Long CoT Supervised Fine-Tuning (SFT) on rejection-sampled NPH graph instances, which substantially enhances reasoning depth, and (ii) Reinforcement Learning (RL) with a fine-grained reward design, which sharpens reasoning efficiency. Our flagship model, **NPG-Muse-7B** , surpasses QwQ-32B on NPH graph problems in both accuracy and reasoning efficiency, and demonstrates strong generalization across mathematics, coding, logic, and STEM. These results position NPH graph problems as an effective and scalable resource for advancing Long CoT reasoning in LLMs, opening a new frontier for LLM post-training.

## Models and Resources

### 🎯 Models

| Model         | Base                   | Parameters | Description          |
| ------------- | ---------------------- | ---------- | -------------------- |
| NPG-Muse-7B   | Qwen2.5-7B-Instruct-1M | 7.62B      | Main model |
| NPG-Muse-1.5B | Qwen2.5-1.5B           | 1.78B      | Lightweight version  |

### 🏢 Organization

Anonymous Research Organization

</div>

## Quick Start on Training

### Data Preparation

Before starting training, you need to download and prepare the data files. These files are large and have been excluded from the repository to keep it lightweight.

#### Download Source Data

```bash
# install toolkit for downloading
pip install gdown

# Download training data
gdown 1SAnriIwjWD3_q9H-4XBJqba_J0luhE0V --folder
# Download testing data
gdown 1JnWDXWUg2gcgfeTqOj7RcUREQ1mtdcjk --folder

# Download relevant data sources
mkdir -p verl/verl/utils/reward_score/tasks/source
cd verl/verl/utils/reward_score/tasks/source
gdown 1meKois5K3SVfTlEhn1FQNfXzq2S6NFvq
tar -xzf source.tar.gz
rm source.tar.gz
```

### Training Environment Setup

Please Follow Verl Setting Shown in `verl/README.md`.

### Two-Stage Training Pipeline

#### Stage 1: Supervised Fine-Tuning

We recommend VERL's SFT framework for **3x speedup** over the original `360-llama-factory` approach.

```bash
cd verl/recipe/NPG-Muse/sft/
bash run_sft.sh
```

#### Stage 2: Reinforcement Learning with Curriculum Learning

```bash
cd verl/recipe/NPG-Muse/rl/
bash run_rl.sh
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
│           │   └── SFT_README.md            #   Documentation
|           ├──rl/                           # Stage 2: RL Training
│           │   ├── run_rl.sh                #   Training script
│           │   └── RL_README.md             #   Documentation
├── eval/                                    # Evaluation scripts
```

## Hardware Requirements

### Recommended Configuration

| Component | Specification |
| --------- | ------------- |
| **GPU**   | 8x A800 80GB  |
