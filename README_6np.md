## 如果卡够多：一次性跑完所有settings，时间短
### 1. 先基于qwen3-8b-base，尝试各种训练settings
> 训练脚本参见train_ablation.sh。为了节约时间，应根据资源情况改为并行执行，并且将6np_curriculum.sh和6np_mixed.sh里面的nnodes和n_gpus_per_node也做相应修改。
### 2. 测试1.得到的各个模型的性能，并基于此选出最佳setting。
> 测以下benchmarks，测完后交给yuyao对比结果和选择setting。
- In-task: GraphArena(NP)
- Cross-Task, In-domain: GraphArena(P), GraphWiz, Node/Link Prediction 
- Cross-Domain (challenging mathematics, pass@k): aime25 (pass@64) / aime24 (pass@64)/ math500 (pass@8)
/ lmb (pass@8) / minerva_math (pass@8) /gaokao2023en(pass@8)/
gaokao2024_mix(pass@8)
- Cross-Domain (all types, pass@1): 
    - (math and stem) college_math, cmath, gsm8k, svamp, asdiv, mawps, sat_math, aqua, tabmwp, amc23, mmlu_stem
    - (logic) zebralogic
    - (code) CLRS
> 以下命令默认串行测试所有模型，如果要一个个测试需要手动修改 bash里的models参数
1. GraphArena
```bash
cd npg_muse_attachments/evaluation/fast/grapharena_evaluation
bash eval_grapharena_all.sh ../../../rl_models
```
2. GraphWiz
```bash
cd npg_muse_attachments/evaluation/fast/graph_realworld_evaluation
bash eval_graph_realworld.sh ../../../rl_models
```
3. Math_pass@k
```bash
cd npg_muse_attachments/evaluation/fast/math_and_stem_evaluation
bash eval_passk.sh ../../../rl_models
```

4. Math_pass@1
```bash
cd npg_muse_attachments/evaluation/fast/math_and_stem_evaluation
bash eval_pass1.sh ../../../rl_models
```

5. Logic

6. CLRS


### 3. 将最佳setting推广到其余base models上，再测试性能。
> 训练脚本参见train_all_series.sh；为了节约时间应该根据资源情况改为并行执行。
## 如果卡不够多：需要逐个消融，时间长
### 1. 固定binary reward + mixed training，消融数据量（45000/60000/90000）
> 动机：6NP相比3NP，任务数量扩增了一倍，因此需要重新考虑数据量问题。单个任务数据量不够难以刷点，总数据量太大又可能导致泛化能力变差。
- 首先同时运行以下几个bash：
```bash
bash recipe/NPG-Muse/rl/choose_data_volume_binary.sh 45000
bash recipe/NPG-Muse/rl/choose_data_volume_binary.sh 60000
bash recipe/NPG-Muse/rl/choose_data_volume_binary.sh 90000
```
- 然后将上述checkpoints转换成models
> 脚本末尾已经进行转换，理论上模型会出现在`./npg_muse_attachments/rl_models`中。如果没找到模型，需要手动修改并运行scripts/merge_model.sh进行转换。
- 将models用于GraphArena/Node&Link/Mathematics的测试（见`./npg_muse_attachments/evaluation/fast`中的代码），找到in-domain和cross-domain平衡得最好的数据量，后续实验同一采用。
### 2. 固定数据量 + binary reward，消融curriculum learning/mixed training
> 动机：之前的curriculum learning限制了response length，现在改为全程8192，看到底curriculum learning对效果是好是坏，是否需要删除。
- 首先运行以下bash
```bash
# 需要用1.中的最佳数据量替换此处的BEST_DATA_VOLUME
bash recipe/NPG-Muse/rl/compare_curriculum_learning_binary.sh $BEST_DATA_VOLUME
```
- 然后将上述checkpoints转换成models（如果能在`./npg_muse_attachments/rl_models`中找到，则不用手动转换）
- 将models用于GraphArena/Node&Link/Mathematics的测试，对比出更好的setting

### 3. 基于curriculum learning / mixed training 中效果更好的setting，消融fine-grained reward
> 动机：对fine-grained reward做完整消融，同时简化了solution quality reward的形式（完全按照比值，丢弃平方项，避免被judge）

- 同时运行以下bash，其中$BEST_STRATEGY应替换为mixed或curriculum
```bash
# binary_format
bash recipe/NPG-Muse/rl/6np_$BEST_STRATEGY.sh npg_muse_attachments/sft_models/qwen3_sft_6np binary_format $BEST_DATA_VOLUME
```
```bash
# binary_format_repeat
bash recipe/NPG-Muse/rl/6np_$BEST_STRATEGY.sh npg_muse_attachments/sft_models/qwen3_sft_6np binary_format_repeat $BEST_DATA_VOLUME
```

```bash
# ratio_quality_format_repeat 
bash recipe/NPG-Muse/rl/6np_$BEST_STRATEGY.sh npg_muse_attachments/sft_models/qwen3_sft_6np ratio_quality_format_repeat $BEST_DATA_VOLUME
```

```bash
# complicated
bash recipe/NPG-Muse/rl/6np_$BEST_STRATEGY.sh npg_muse_attachments/sft_models/qwen3_sft_6np complicated $BEST_DATA_VOLUME
```
- 然后将上述checkpoints转换成models（如果能在./npg_muse_attachments/rl_models中找到，则不用手动转换）
- 将models用于GraphArena/Node&Link/Mathematics的测试，对比出更好的setting
## 4. 基于最好的reward setting，重新对比curriculum learning/mixed training
> 动机：消融实验需要用到这个结果；结论大概率跟2.一致，只是具体数据会不同。

运行以下bash，其中BEST_REWARD为3.中效果最好的，为其中之一：
- binary
- binary_format
- binary_format_repeat
- ratio_quality_format_repeat
- complicated
```bash
bash recipe/NPG-Muse/rl/6np_$BEST_STRATEGY.sh npg_muse_attachments/sft_models/qwen3_sft_6np $BEST_REWARD $BEST_DATA_VOLUME
```
### 5. 基于上述最好的RL setting，在不同模型上训练
> 动机：证明对不同series models的有效性，并说明跟G1是parallel的工作。

同时运行以下命令
```bash
bash recipe/NPG-Muse/rl/6np_$BEST_STRATEGY.sh npg_muse_attachments/sft_models/qwen25_1m_sft_6np $BEST_REWARD $BEST_DATA_VOLUME
```
```bash
bash recipe/NPG-Muse/rl/6np_$BEST_STRATEGY.sh npg_muse_attachments/sft_models/llama31_sft_6np $BEST_REWARD $BEST_DATA_VOLUME
```
```bash
bash recipe/NPG-Muse/rl/6np_$BEST_STRATEGY.sh npg_muse_attachments/sft_models/g1_sft_6np $BEST_REWARD $BEST_DATA_VOLUME
```