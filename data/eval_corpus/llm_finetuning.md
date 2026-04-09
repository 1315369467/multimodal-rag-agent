# LLM大模型微调技术全面指南

## 第一章 微调方法概述

### 1.1 为什么需要微调

预训练大模型（如LLaMA、Qwen）具备通用语言理解能力，但在以下场景中需要微调：
- **领域适应**：法律、医疗、金融等专业领域的术语和推理模式
- **任务对齐**：将通用能力转化为特定任务（分类、信息抽取、问答等）
- **格式控制**：让模型按照固定格式输出JSON、Markdown等结构化内容
- **价值对齐**：通过RLHF/DPO等使模型更符合人类偏好

**微调方法分类**：
| 方法 | 参数更新量 | 显存需求 | 典型用途 |
|------|----------|---------|---------|
| 全量微调（Full Fine-tuning） | 100% | 极高（模型×3-4倍） | 大规模数据，最优效果 |
| LoRA | 0.1-1% | 低（+额外少量） | 最常用，效果接近全量 |
| QLoRA | 0.1-1% | 极低（量化基模型） | 消费级GPU微调大模型 |
| Prefix Tuning | <0.1% | 极低 | 任务迁移 |
| Prompt Tuning | <0.01% | 极低 | 少量任务适配 |
| 适配器（Adapter） | <1% | 低 | 多任务共享基模型 |

---

### 1.2 全量微调（Full Fine-tuning）

所有模型参数均参与更新，效果最好，但资源需求最高。

**显存需求估算**（以Adam优化器为例）：
- 模型权重：参数量 × 2字节（float16）
- 梯度：参数量 × 2字节
- Adam一阶矩（m）：参数量 × 4字节（float32）
- Adam二阶矩（v）：参数量 × 4字节（float32）
- **总计：参数量 × 12字节**，即7B模型约需84GB显存

**实用建议**：
- 使用bf16混合精度训练，通过gradient checkpointing将激活值显存降低约60%
- DeepSpeed ZeRO-3可将显存需求分片到多卡，理论上可将单卡显存需求降至1/N
- 7B模型全量微调推荐至少4×A100 80GB

---

## 第二章 LoRA与参数高效微调（PEFT）

### 2.1 LoRA原理

LoRA（Low-Rank Adaptation，低秩适配）是目前最广泛使用的微调方法，核心思想是：**预训练权重的更新矩阵 $\Delta W$ 具有低秩特性**，可以分解为两个小矩阵的乘积。

**数学形式**：
$$W' = W + \Delta W = W + \frac{\alpha}{r} BA$$

- $W \in \mathbb{R}^{d \times k}$：冻结的预训练权重
- $A \in \mathbb{R}^{r \times k}$：下投影矩阵，随机高斯初始化
- $B \in \mathbb{R}^{d \times r}$：上投影矩阵，初始化为零（保证训练初始与原模型等价）
- $r$：秩（rank），通常取4、8、16、64
- $\alpha$：缩放因子，通常等于r或r的一半

**参数效率**：额外参数量为 $r \times (d + k)$，而原始矩阵为 $d \times k$。当 $r \ll \min(d,k)$ 时，参数量减少极显著。

**典型参数**：7B模型中，每个注意力层 $d=4096,k=4096$，LoRA r=8时只需 $8 \times (4096+4096) = 65536$ 个参数，而原矩阵有 $4096 \times 4096 = 16M$ 个参数，**参数量减少99.6%**。

### 2.2 LoRA关键超参数

**rank（r）**：
- r=4：极度参数高效，适合数据量少（<1K条）的场景
- r=8：最常用默认值，平衡效率与效果
- r=16/32：复杂任务或数据量较大时使用
- r=64/128：接近全量微调效果，适合高资源场景

**target_modules（目标模块）**：
- 最小配置：仅微调注意力中的Q和V矩阵（`["q_proj", "v_proj"]`）
- 推荐配置：所有注意力矩阵（Q、K、V、O）加上FFN（`["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`）
- 实验结论：加入FFN的gate_proj/up_proj/down_proj通常可将效果提升5-10%

**lora_alpha**：控制LoRA更新的缩放幅度，通常设为rank的1-2倍。alpha越大，微调更新越激进，学习率等效越高。

**lora_dropout**：正则化手段，通常设为0.05-0.1。数据量较少时适当提高。

### 2.3 QLoRA

QLoRA（Quantized LoRA）将基础模型量化为4bit存储，在此基础上添加float16精度的LoRA适配器进行训练。

**核心技术**：
- **NF4（Normal Float 4-bit）**：一种信息理论上最优的4bit量化格式，比INT4精度更高
- **双重量化（Double Quantization）**：对量化常数本身再量化，额外节省约0.37bit/参数
- **分页优化器（Paged Optimizers）**：用CPU内存分页存储优化器状态，避免显存OOM

**资源对比**：
| 方法 | LLaMA 2 13B所需显存 | LLaMA 2 70B所需显存 |
|------|-----------------|-----------------|
| 全量微调 | ~156GB | ~840GB |
| LoRA（fp16） | ~28GB | ~140GB |
| QLoRA（4bit） | ~10GB | ~48GB |

QLoRA使得在**单张24GB消费级GPU上微调33B模型**成为可能，极大降低了微调门槛。

### 2.4 LoRA变体

**LoRA+**：对B矩阵使用更大的学习率（通常B:A = 16:1），加速收敛，效果优于标准LoRA约1-2%。

**DoRA（权重分解LoRA）**：将权重分解为幅度（magnitude）和方向（direction）两部分，分别优化，更接近全量微调的学习动态，在多数任务上优于标准LoRA。

**LoRA-FA（冻结A矩阵）**：固定A矩阵的随机初始化，只训练B矩阵，显存节省约30%，效果略有下降。

**VeRA**：所有层共享同一个随机初始化的A、B矩阵，只学习少量缩放向量，参数量减少100倍以上，适合超低资源场景。

---

## 第三章 指令微调与对齐训练

### 3.1 指令微调（Instruction Tuning/SFT）

**数据格式**：指令微调通常采用Chat Template格式，不同模型有不同模板。

**LLaMA 2 Chat格式**：
```
[INST] <<SYS>>
{system_prompt}
<</SYS>>
{user_message} [/INST] {assistant_response}
```

**Qwen格式**：
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
```

**数据质量原则**（Lima论文结论）：
- 1000条高质量指令数据可达到与10万条普通数据相当甚至更优的效果
- 数据多样性比数量更重要：覆盖不同任务类型、领域、难度
- 过滤低质量样本：去重、去除错误答案、去除格式混乱的样本

**只计算response部分的Loss**：在SFT训练中，通常只对assistant的回复部分计算交叉熵损失，对system和user部分设置为-100（忽略），避免模型学习"复述用户输入"。

### 3.2 RLHF（人类反馈强化学习）

RLHF是ChatGPT、Claude等对话模型实现价值对齐的核心技术，分三个阶段：

**阶段一：SFT（有监督微调）**
使用人工编写的高质量对话数据进行有监督训练，初步建立指令遵循能力。

**阶段二：奖励模型（Reward Model, RM）训练**
对同一个输入收集多个不同质量的回复，由人工标注偏好排序（A > B > C），训练一个奖励模型来预测人类偏好分数。

**阶段三：PPO强化学习**
使用PPO（近端策略优化）算法，以奖励模型的分数作为奖励信号，通过强化学习进一步优化SFT模型，同时引入KL散度惩罚项防止模型偏离SFT基础过远：
$$r = r_{RM}(x, y) - \beta \cdot \text{KL}[\pi_\theta || \pi_{SFT}]$$

**RLHF的挑战**：
- 奖励模型本身可能有偏，导致**奖励欺骗（Reward Hacking）**
- 训练不稳定，超参数敏感
- 人工标注成本高，规模化困难

### 3.3 DPO（直接偏好优化）

DPO是RLHF的简化替代方案，无需训练独立的奖励模型，直接从偏好数据中优化策略。

**DPO损失函数**：
$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

- $y_w$：偏好回复（preferred/chosen）
- $y_l$：被拒绝回复（rejected）
- $\pi_{ref}$：参考模型（通常是SFT模型）
- $\beta$：控制偏离参考模型的程度，通常取0.1-0.5

**DPO的优势**：
- 无需强化学习，训练稳定，代码实现简单
- 显存需求与SFT相同
- 在多数任务上效果与RLHF相当甚至更优

**DPO数据格式**：每条样本需要包含 `(prompt, chosen, rejected)` 三元组，来源可以是人工标注、GPT-4打分、或AI反馈（RLAIF）。

---

## 第四章 微调实践与常见问题

### 4.1 数据准备

**数据量参考**：
| 任务类型 | 推荐数据量 | 说明 |
|---------|----------|------|
| 特定格式输出 | 500-2000条 | 模型已有基础能力，只需格式适配 |
| 单一领域问答 | 3000-10000条 | 需要注入领域知识 |
| 通用指令遵循 | 10000-100000条 | 全面提升指令遵循能力 |
| 预训练领域适配 | 1B+ tokens | 大规模继续预训练 |

**数据质量检查清单**：
- 去重：使用MinHash LSH检测近似重复，去除相似度>0.9的样本
- 长度过滤：去除token数<10或超过模型最大上下文的样本
- 语言过滤：确保目标语言占比
- 格式验证：JSON输出任务需验证JSON合法性
- 多样性分析：统计任务类型分布，避免过度集中

### 4.2 训练超参数设置

**学习率**：
- SFT（基于预训练模型）：$1e^{-5}$ 至 $3e^{-5}$
- SFT（基于指令模型）：$5e^{-6}$ 至 $1e^{-5}$，更小以防止遗忘
- LoRA：$1e^{-4}$ 至 $3e^{-4}$（LoRA参数学习率通常比全量高一个量级）
- Warmup比例：通常为总步数的3-5%

**批次大小**：
- 有效批次大小（Effective Batch Size）= 单卡Batch × 梯度累积步数 × GPU数量
- 推荐有效批次大小：128-512条（过小会导致训练不稳定）
- 显存不足时优先考虑梯度累积而非减小batch size

**训练轮次**：
- 通常1-3个epoch，过多会导致遗忘预训练知识（Catastrophic Forgetting）
- 使用验证集监控loss，出现过拟合时提前停止

### 4.3 常见问题排查

**问题1：训练损失下降但指标不改善**
- 原因：评估方式与训练目标不一致，或数据质量问题
- 解决：检查数据标注质量；对比LoRA微调前后的输出差异；确保评估prompt格式与训练一致

**问题2：灾难性遗忘（Catastrophic Forgetting）**
- 症状：微调后原有能力（如代码生成、数学推理）显著下降
- 原因：微调数据与预训练数据分布差异过大，学习率过高
- 解决：降低学习率；混入通用数据（10-30%）；优先选择LoRA而非全量微调；使用更小的epoch数

**问题3：模型输出格式不稳定**
- 症状：有时输出符合要求的格式，有时不符合
- 原因：数据量不足，或指令中格式要求不明确
- 解决：增加格式示例数据（few-shot）；使用更明确的系统提示；考虑输出层的约束解码

**问题4：中文能力下降**
- 原因：使用英文为主的指令数据微调后，中文能力被"稀释"
- 解决：保持中英文数据比例与预训练阶段接近；中文应用优先使用Qwen等中文友好基模型

**问题5：LoRA微调效果不如预期**
- 检查target_modules是否包含了足够的模块（建议加入FFN层）
- rank是否足够（复杂任务尝试r=64）
- 检查chat template是否正确应用（LLaMA和Qwen模板不同）
- 确认是否正确mask了非assistant部分的loss

### 4.4 微调后评估

**自动评估**：
- 困惑度（Perplexity）：衡量语言建模质量，但与任务表现相关性有限
- 任务特定指标：ROUGE（摘要）、BLEU（翻译）、Exact Match（问答）、F1（信息抽取）
- LLM-as-Judge：使用GPT-4对微调模型输出进行打分（1-5分），覆盖流畅性、准确性、有用性

**人工评估（AB测试）**：
- 盲评：不告知评估者哪个是微调版本
- 评分维度：准确性、有用性、安全性、格式规范性
- 评估样本：至少200条，覆盖全部目标场景

**回归测试**：
- 维护一个通用能力测试集，确保微调后不退化
- 重点检查：数学推理（GSM8K）、代码生成（HumanEval）、通用问答（MMLU）

---

## 第五章 微调框架与工具

### 5.1 主流框架对比

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| LLaMA-Factory | 功能最全面，支持100+模型，WebUI友好 | 快速实验，非代码用户 |
| Axolotl | 配置灵活，支持多种PEFT方法 | 生产级微调流程 |
| HuggingFace TRL | 官方支持，SFT/DPO/PPO均有实现 | 标准Pipeline |
| Unsloth | 极致速度优化（比标准快2倍），低显存 | 消费级GPU用户 |
| DeepSpeed | 大规模分布式训练，ZeRO优化 | 多机多卡大规模微调 |

### 5.2 LLaMA-Factory关键配置示例

```yaml
# LoRA SFT配置
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
finetuning_type: lora
lora_rank: 8
lora_target: all  # 等价于所有线性层
lora_alpha: 16

dataset: alpaca_zh  # 数据集名称
template: qwen      # Chat Template类型
cutoff_len: 2048    # 最大序列长度

per_device_train_batch_size: 4
gradient_accumulation_steps: 8  # 有效batch=32
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.05

bf16: true
flash_attn: fa2  # Flash Attention 2
```

### 5.3 合并LoRA权重

训练完成的LoRA适配器需要与基模型合并才能进行高效推理（避免推理时额外计算overhead）：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("base_model_path")
model = PeftModel.from_pretrained(base_model, "lora_adapter_path")
merged_model = model.merge_and_unload()  # 合并LoRA权重
merged_model.save_pretrained("merged_model_path")
```

合并后的模型大小与原基础模型相同，推理速度无额外开销。

---

## 第六章 微调成本与效益分析

### 6.1 GPU时间与成本估算

| 模型 | 方法 | 数据量 | GPU配置 | 估算时间 | 云端成本（A100） |
|------|------|-------|---------|---------|--------------|
| 7B | QLoRA | 10K条 | 1×A100 40G | ~2小时 | ~$6 |
| 7B | LoRA | 50K条 | 1×A100 80G | ~8小时 | ~$24 |
| 13B | QLoRA | 10K条 | 1×A100 80G | ~4小时 | ~$12 |
| 70B | QLoRA | 10K条 | 2×A100 80G | ~12小时 | ~$72 |
| 7B | 全量 | 50K条 | 4×A100 80G | ~6小时 | ~$72 |

**成本优化建议**：
- 优先使用QLoRA，尤其是资源有限时
- 使用Spot/Preemptible实例可节省50-70%费用（需要断点续训支持）
- 中小规模任务优先选7B模型，效果不足再升级到13B/70B

### 6.2 效果收益预期

基于工业实践数据：
- **领域知识注入**：专业领域问答准确率从40-60%提升至75-85%
- **格式控制**：结构化输出符合率从60-70%提升至95%以上
- **中文指令遵循**：中文任务Score从基础模型的70-75分提升至85-90分（满分100）
- **与GPT-4 API方案对比**：7B本地微调模型在特定垂直任务上可达到GPT-4的80-90%效果，但成本降低95%以上
