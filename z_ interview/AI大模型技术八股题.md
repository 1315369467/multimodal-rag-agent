# AI 大模型技术八股题

> 覆盖 LLM/VLM 核心架构、预训练、监督微调（SFT）、强化学习对齐（PPO、DPO、GRPO），以及 Qwen、DeepSeek 等主流模型技术方案。

---

## 一、Transformer 与 LLM 基础架构

**Q：Transformer 的核心机制是什么？**

Transformer 的核心是**自注意力机制（Self-Attention）**。对输入序列的每个 token，计算它和序列中所有其他 token 的关联权重，然后用这些权重对 Value 向量做加权求和，得到融合了全局上下文的表示。

具体来说，每个 token 被投影成 Q（Query）、K（Key）、V（Value） 三个向量，注意力分数 = softmax(QK^T / √d_k) · V。除以 √d_k 是为了防止点积值过大导致 softmax 梯度消失。

和 RNN 相比，Self-Attention 可以直接建模任意两个位置之间的依赖关系，不存在长距离衰减问题，而且计算可以并行化。

---

**Q：Multi-Head Attention 为什么要用多头？**

单个 Attention Head 只能学习一种注意力模式，多头机制把 Q、K、V 分别投影到 h 个不同的子空间，每个 Head 独立做注意力计算，最后把 h 个输出拼接起来过一个线性层。

好处是不同 Head 可以学习不同类型的依赖关系：有的 Head 可能关注局部语法结构，有的关注长距离语义关联，有的关注位置信息。多头相当于让模型从多个角度同时"看"输入序列，表达能力更强。

参数量和单头 Attention 基本一样（因为每个 Head 的维度是 d_model / h），但效果明显好。

---

**Q：Encoder-only、Decoder-only、Encoder-Decoder 三种架构各有什么特点？**

**Encoder-only（如 BERT）**：双向注意力，每个 token 能同时看到左右两边的内容。适合理解类任务（分类、NER、句子相似度），不擅长生成。预训练用 Masked Language Model（随机遮住 token 让模型预测）。

**Decoder-only（如 GPT、Qwen、DeepSeek）**：因果注意力（Causal Attention），每个 token 只能看到左边的内容，通过 causal mask 实现。适合生成类任务，也是目前大模型的主流架构。预训练用 Next Token Prediction。

**Encoder-Decoder（如 T5、BART）**：编码器双向处理输入，解码器自回归生成输出，中间通过 Cross-Attention 连接。适合 seq2seq 任务（翻译、摘要），但参数量翻倍，训练和推理效率不如纯 Decoder-only。

目前业界几乎都选 Decoder-only，原因是：统一的自回归目标更容易 scaling，预训练数据利用效率高，而且在 In-Context Learning 上表现突出。

---

**Q：为什么现在主流 LLM 都用 Decoder-only 架构？**

核心原因有三个：

1. **训练效率高**：Next Token Prediction 目标简单统一，每个 token 都参与损失计算（不像 BERT 只有 15% 被 mask 的 token 算 loss），数据利用率高
2. **Scaling Law 友好**：随着参数量和数据量增大，Decoder-only 模型的 loss 平滑下降，表现可预测，便于做大
3. **In-Context Learning 涌现**：Decoder-only 架构在达到一定规模后涌现出了强大的 Few-shot 能力，给几个示例就能完成新任务，这是 Encoder-only 模型做不到的

此外，Decoder-only 在工程上也更简单——只有一个模型组件，推理时只需要维护 KV cache，不需要处理编码器-解码器之间的 Cross-Attention。

---

**Q：什么是位置编码，RoPE 的原理是什么？**

Transformer 的 Self-Attention 本身是**位置无关**的，打乱输入 token 的顺序不会改变输出（因为是全局加权求和）。位置编码的作用就是把位置信息注入模型，让模型知道每个 token 在序列里的位置。

早期 Transformer 用绝对位置编码（正弦函数或可学习 embedding），但外推性差——训练长度之外的位置没有见过。

**RoPE（Rotary Position Embedding）** 是目前主流方案（Qwen、DeepSeek 都在用）。核心思路是把位置信息编码为旋转角度，对 Q 和 K 向量做旋转变换。两个 token 做 attention 时，它们的相对位置信息自然地体现在 Q·K 的点积里。

RoPE 的优点：
- **天然编码相对位置**：注意力分数只取决于两个 token 的距离，不依赖绝对位置
- **外推性好**：配合 NTK-aware 插值等方法，可以在推理时扩展到比训练更长的序列
- **对长上下文友好**：远距离的注意力分数会自然衰减（旋转角度变大），类似于一种隐式的位置衰减

---

**Q：LayerNorm 放在哪里有讲究吗？Pre-Norm vs Post-Norm？**

**Post-Norm（原始 Transformer）**：先做 Attention/FFN，再做 LayerNorm。训练不稳定，尤其在模型很深的时候容易梯度爆炸，需要精心调学习率和 warmup。

**Pre-Norm（现在主流）**：先做 LayerNorm，再做 Attention/FFN。训练更稳定，不需要特别的学习率调度，收敛更快。几乎所有现代大模型（GPT、Qwen、LLaMA、DeepSeek）都用 Pre-Norm。

此外，很多模型把 LayerNorm 替换成了 **RMSNorm**（Root Mean Square Normalization），去掉了均值中心化步骤，只做方差归一化，计算更快，效果差不多。

---

**Q：什么是 GQA（Grouped Query Attention），为什么要用？**

标准的 Multi-Head Attention 里，每个 Head 有独立的 Q、K、V。KV cache 的大小和 Head 数成正比，序列长了之后 KV cache 非常占显存。

**MQA（Multi-Query Attention）**：所有 Head 共享同一组 K 和 V，只有 Q 是独立的。KV cache 减少到 1/h，推理速度快了很多，但质量下降比较明显。

**GQA（Grouped Query Attention）**：折中方案。把 h 个 Head 分成 g 组（比如 32 个 Head 分成 8 组），组内共享 K 和 V。KV cache 减少到 1/(h/g)，在推理效率和质量之间取得平衡。

Qwen2、LLaMA 2（70B 版本）、DeepSeek 都用了 GQA。

---

**Q：FFN（前馈网络）在 Transformer 里起什么作用？**

Self-Attention 负责建模 token 之间的关系，FFN 负责对每个 token 的表示做非线性变换，起到**特征提取和记忆存储**的作用。

研究表明，Transformer 的知识主要存储在 FFN 的权重矩阵里（可以类比为 key-value memory），而 Attention 层更多起路由和信息聚合的作用。

现代大模型通常用 **SwiGLU** 激活函数替代原始的 ReLU：`SwiGLU(x) = Swish(xW₁) ⊙ (xV)`，引入了门控机制，效果比 ReLU 和 GELU 都好，是目前的标配（Qwen、LLaMA、DeepSeek 都在用）。

---

## 二、大模型预训练

**Q：LLM 预训练的目标是什么？**

Decoder-only LLM 的预训练目标是 **Next Token Prediction（NTP）**：给定前面的 token 序列 x₁...xₜ，预测下一个 token xₜ₊₁ 的概率分布，最小化交叉熵损失。

这个目标看似简单，但要在海量语料上准确预测下一个 token，模型必须学会语法、语义、常识、逻辑推理等各种能力——**语言建模是一种通用的无监督学习目标**。

预训练数据通常是数万亿 token 的互联网文本、书籍、代码等混合语料。数据质量和配比对模型效果影响极大，通常需要大量数据清洗和去重。

---

**Q：Scaling Law 是什么，有什么实际意义？**

Scaling Law（Kaplan et al., 2020 / Chinchilla, 2022）揭示了模型性能和三个变量的关系：**参数量 N、训练数据量 D、计算量 C**。

核心发现：
- 模型 loss 随 N、D、C 的增长呈**幂律下降**，可以用简单的公式预测
- 三者之间存在最优分配比例：给定计算预算，参数量翻倍时训练数据量也应该翻倍（Chinchilla 法则）
- 之前很多模型"参数多、数据少"，实际上是 undertrained

实际意义：可以用小规模实验预测大模型的最终效果，指导训练资源分配。比如在 1B 模型上跑几组消融实验，就能预估 70B 模型需要多少数据、大概什么 loss，避免浪费算力。

---

**Q：预训练数据处理有哪些关键步骤？**

主要步骤：

1. **数据收集**：互联网爬取（Common Crawl）、书籍、论文、代码（GitHub）、百科等多源混合
2. **去重**：MinHash 或 SimHash 做近似去重，移除高度重复的文本。重复数据会导致模型在特定模式上过拟合
3. **质量过滤**：用分类器或启发式规则过滤低质量内容（广告、乱码、SEO 垃圾），部分模型会用小型 LM 做质量打分
4. **敏感内容过滤**：移除有害内容、个人隐私信息
5. **数据配比**：不同来源的数据按一定比例混合，代码数据比例大会提升推理能力，但太多会降低自然语言能力
6. **Tokenization**：用 BPE（Byte Pair Encoding）或 SentencePiece 把文本切分成 subword token，构建词表

---

**Q：什么是 BPE（Byte Pair Encoding），为什么不直接用字或词？**

直接用字（character-level）的问题是序列太长，训练和推理效率低。直接用词（word-level）的问题是词表巨大（几十万），而且无法处理未登录词（OOV）。

**BPE** 是一种折中方案：从字符级开始，统计相邻字符对的出现频率，反复合并最高频的字符对，直到词表达到预设大小（比如 32000、100000）。

结果是：高频词被当作整体（比如 "the"），低频词被拆成子词（比如 "unhappiness" → "un" + "happiness"），极低频词退化为字符级。这样词表大小可控，又能处理任意输入。

现代模型通常用 **Byte-level BPE**（GPT 系列）或 **SentencePiece**（Qwen、LLaMA），本质类似。

---

**Q：预训练的分布式训练策略有哪些？**

大模型预训练动辄几百上千张 GPU，需要多种并行策略组合：

**数据并行（Data Parallelism）**：每张 GPU 持有完整模型副本，各自处理不同数据，梯度做 AllReduce 同步。简单有效，但每张卡要放下完整模型。ZeRO 优化把优化器状态、梯度、参数分片到多卡，大幅降低单卡显存。

**张量并行（Tensor Parallelism）**：把单层的矩阵运算切分到多张 GPU 上。比如一个 FFN 的权重矩阵按列切成 N 份，每张 GPU 算一部分，再做 AllReduce。适合单机多卡，通信密集。

**流水线并行（Pipeline Parallelism）**：把模型的不同层分到不同 GPU 上，数据像流水线一样流过。减少单卡的显存和计算量，但有 bubble time（有些卡在等上游数据）。

**序列并行（Sequence Parallelism）**：把长序列切分到多卡处理，解决长上下文训练的显存瓶颈。

实际训练中通常组合使用，比如 DeepSeek-V3 用了 ZeRO + Pipeline + Tensor 的混合并行方案。

---

## 三、监督微调（SFT）

**Q：什么是 SFT，和预训练有什么区别？**

**预训练**是在海量无标注文本上学习通用语言能力，目标是 Next Token Prediction，模型学到的是"世界是什么样的"。

**SFT（Supervised Fine-Tuning）** 是在高质量的**指令-回答对**上继续训练，让模型学会"如何遵循指令回答问题"。数据格式通常是 `(instruction, input, output)`，损失只算在 output 部分（不回传 instruction 的梯度）。

SFT 的作用：
- 把预训练模型从"续写文本"变成"回答问题"
- 注入特定格式和风格（比如总是用 Markdown 格式回答）
- 对齐特定领域知识（比如医疗、法律问答）

SFT 数据量通常很小（几千到几万条），但质量极其重要——几百条高质量数据 > 几万条低质量数据。

---

**Q：SFT 数据怎么构建，有什么注意事项？**

构建方式：

1. **人工标注**：领域专家编写高质量 QA 对，质量最高但成本大
2. **Self-Instruct**：用强模型（如 GPT-4）根据种子指令生成更多指令和回答，再人工筛选
3. **蒸馏**：用强模型对同一批问题生成答案，作为弱模型的训练数据

注意事项：
- **多样性**比数量重要：覆盖不同任务类型、不同难度、不同领域
- **答案质量**直接决定模型上限：一条错误的高权重样本能拉低整体表现
- **格式一致**：prompt 模板要和推理时保持一致（chat template），不一致会导致效果下降
- **Loss mask**：只在 output 部分算 loss，instruction 部分的 token 不参与梯度更新

---

**Q：全参数微调和 LoRA 有什么区别？**

**全参数微调**：更新模型所有参数，效果最好，但显存需求大（需要存储完整的梯度和优化器状态），而且容易在小数据集上过拟合，灾难性遗忘风险高。

**LoRA（Low-Rank Adaptation）**：冻结原始模型参数，在每个目标层（通常是 Attention 的 Q、K、V 投影层）旁边插入两个低秩矩阵 A（d×r）和 B（r×d），r 远小于 d（比如 r=16，d=4096）。训练时只更新 A 和 B，参数量只有原始的 0.1%-1%。

LoRA 的核心假设是**微调时的权重变化是低秩的**——大模型在预训练时已经学到了足够好的表示，微调只需要做小幅度的方向调整，不需要改变全部参数。

推理时 LoRA 权重可以合并回原始权重：W' = W + BA，不增加额外推理开销。

---

**Q：LoRA 的秩 r 怎么选？**

r 是 LoRA 最核心的超参，控制可训练参数量和表达能力：

- **r 太小**（比如 4）：参数量极少，表达能力受限，可能欠拟合，适合任务简单、数据少的场景
- **r 太大**（比如 256）：接近全参微调，失去 LoRA 的效率优势，而且小数据上容易过拟合
- **常用范围**：8-64，通常 16 或 32 是一个不错的起点

选择策略：
- 任务和预训练分布差距大 → r 大一些
- 数据量少 → r 小一些
- 先用 r=16 跑 baseline，再根据效果调整

还有 **QLoRA**：在 LoRA 基础上把原始模型量化到 4-bit 再训练 LoRA，显存进一步减少，可以在单张消费级 GPU 上微调 7B 甚至 13B 模型。

---

## 四、强化学习对齐（RLHF）

**Q：什么是对齐（Alignment），为什么需要？**

经过预训练和 SFT 的模型已经能回答问题了，但它可能会输出有害内容、不符合人类价值观的回答、或者明知不对还一本正经地编。

**对齐（Alignment）** 的目标是让模型的输出和人类的偏好一致：有用、诚实、无害。

核心矛盾是：SFT 只能教模型"什么样的回答格式是对的"，但很难教它"两个回答哪个更好"——这种偏好信息很难用 (instruction, output) 的形式表达。强化学习对齐的思路是引入**人类偏好反馈**，让模型学习"哪些行为人类更喜欢"。

---

**Q：RLHF 的三个阶段分别做什么？**

经典的 RLHF（Reinforcement Learning from Human Feedback）流程分三步：

**阶段一：SFT**
在高质量指令数据上做监督微调，得到一个能正常回答问题的初始策略模型。

**阶段二：训练奖励模型（Reward Model）**
收集人类偏好数据——同一个问题让模型生成多个回答，人类标注员对这些回答排序（哪个更好）。用这些偏好对训练一个奖励模型，输入一个 (prompt, response) 对，输出一个标量分数，表示这个回答有多好。

**阶段三：PPO 强化学习**
用 PPO 算法优化策略模型（SFT 后的模型），最大化奖励模型给出的分数，同时加一个 KL 散度约束，防止模型偏离 SFT 阶段的分布太远（避免 reward hacking）。

---

**Q：PPO 算法的核心思路是什么？**

PPO（Proximal Policy Optimization）是一种策略梯度算法，核心思路是**限制每次更新的步幅**，避免策略变化太大导致训练崩溃。

在 RLHF 场景里：
- **策略（Policy）**：就是 LLM 本身，给定 prompt 生成 response
- **奖励（Reward）**：奖励模型对 response 的评分
- **优化目标**：最大化期望奖励，同时限制新策略和旧策略的差距

PPO 用了 **clipped surrogate objective**：计算新旧策略的概率比 rₜ = π_new / π_old，如果 rₜ 偏离 1 太远（超出 [1-ε, 1+ε] 范围），就截断梯度，防止一次更新改变太多。

RLHF 里还有一项 KL 惩罚：reward_final = reward_model_score - β · KL(π_new || π_sft)，防止模型为了讨好奖励模型而生成不自然的文本（reward hacking）。

---

**Q：PPO 在 LLM 对齐里有什么问题？**

PPO 效果好但**工程复杂度极高**：

1. **需要四个模型同时在线**：策略模型（Actor）、参考模型（Reference，用于 KL 约束）、奖励模型（Critic）、价值网络（Value），显存需求巨大
2. **训练不稳定**：奖励模型的质量直接决定了 PPO 的上限，如果奖励模型有偏差，PPO 会放大这个偏差（reward hacking）
3. **超参敏感**：学习率、KL 系数 β、clip range ε、batch size 都需要仔细调，不同设置效果差距很大
4. **采样开销大**：每一步都要用当前策略模型生成完整 response，再算 reward，生成过程很慢

这些问题催生了 DPO 等不需要奖励模型的替代方案。

---

**Q：DPO 和 PPO 有什么区别，为什么更简单？**

**DPO（Direct Preference Optimization）** 的核心洞察是：**可以跳过奖励模型，直接从人类偏好数据优化策略模型**。

数学上，DPO 证明了 RLHF 的最优解可以用一个解析形式表达，把 reward model 的 loss 直接转化成策略模型的 loss。最终的训练目标是：给定 (prompt, preferred_response, rejected_response) 三元组，让模型对 preferred response 的概率增大、对 rejected response 的概率减小。

和 PPO 相比，DPO 的优势：
- **不需要奖励模型**：只需要策略模型和参考模型两个，显存减半
- **不需要在线采样**：直接在离线偏好数据上做 supervised 训练，和 SFT 的训练流程几乎一样
- **训练稳定**：没有 RL 的 high variance 问题，超参也更容易调

劣势：
- 离线训练依赖数据分布，如果偏好数据和当前策略分布差距太大，效果会下降
- 没有 PPO 那样的在线探索能力，可能陷入局部最优

---

**Q：GRPO 是什么，和 PPO/DPO 有什么区别？**

**GRPO（Group Relative Policy Optimization）** 是 DeepSeek 提出的强化学习算法，用在 DeepSeek-R1 的训练中。

核心思路：对每个 prompt，让模型采样**一组回答**（比如 64 个），用规则或者奖励模型对每个回答打分，然后**组内相对排名**来计算优势值（advantage），不需要额外的 Value 网络（Critic）。

和 PPO 的区别：
- **去掉了 Critic 模型**：PPO 需要一个 Value 网络估计状态价值来计算 advantage，GRPO 直接用组内回答的相对排名，减少了一个模型的显存开销
- **组内对比**：同一个 prompt 下的多个回答互相比较，自然地归一化了 reward 分布，不需要额外做 reward normalization
- **更适合可验证任务**：数学、代码这类有标准答案的任务可以直接用规则打分（答对/答错），不需要训练奖励模型

和 DPO 的区别：
- GRPO 是**在线**的，用当前策略实时采样并优化，探索能力更强
- DPO 是离线的，依赖预先收集的偏好数据

GRPO 在 DeepSeek-R1 中的应用表明，纯 RL（不经过 SFT）也能让模型涌现出 Chain-of-Thought 推理能力。

---

**Q：奖励模型（Reward Model）怎么训练？**

奖励模型通常基于 SFT 模型初始化，去掉最后的 language model head，换成一个输出标量分数的线性层。

训练数据是**偏好对**：(prompt, response_win, response_lose)，表示人类认为 response_win 比 response_lose 更好。

训练目标是 **Bradley-Terry 模型**：
loss = -log(σ(r(prompt, win) - r(prompt, lose)))

即让 win response 的分数高于 lose response 的分数，margin 越大越好。

注意事项：
- 偏好标注的一致性非常重要，标注员之间的分歧大会导致奖励模型质量差
- 奖励模型的泛化能力是关键，如果只在特定分布的数据上训练，对分布外的输入打分不准，PPO 就会 hack 这些盲点
- 一些最新的做法（如 DeepSeek-R1）用**基于规则的奖励**替代奖励模型，比如数学题直接判断答案对不对，代码题直接跑测试用例

---

## 五、VLM（视觉语言模型）

**Q：VLM 的基本架构是什么？**

VLM 的核心是把**视觉信息注入 LLM**，典型架构有三部分：

1. **视觉编码器（Vision Encoder）**：通常是预训练的 ViT（Vision Transformer），把图片切成 patch 后编码成一组视觉 token
2. **投影层（Projector）**：把视觉 token 的维度映射到 LLM 的维度空间，让 LLM 能理解。常见方式有 MLP、Cross-Attention、Q-Former 等
3. **LLM 骨干**：处理融合后的文本+视觉 token 序列，生成文本输出

不同的 VLM 主要区别在**视觉 token 怎么注入 LLM**：
- LLaVA：最简单，MLP 投影后直接拼接在文本序列里
- Qwen-VL：用 Cross-Attention 在特定层注入
- InternVL：大规模视觉编码器 + MLP 投影

---

**Q：ViT（Vision Transformer）的工作原理是什么？**

ViT 把 CV 任务变成了序列建模问题：

1. 把图像切成固定大小的 patch（比如 14×14 像素）
2. 每个 patch 展平后通过线性投影变成一个向量（类比于 NLP 里的 token embedding）
3. 加上位置编码后，送入标准的 Transformer Encoder
4. 取 `[CLS]` token 或做 mean pooling 作为图像的全局表示

ViT 的优势是可以直接复用 Transformer 的架构和训练方法，而且在大规模数据上效果超过 CNN。缺点是对数据量要求大，小数据上不如 CNN。

在 VLM 里，通常用预训练好的 ViT（如 CLIP-ViT、SigLIP）作为视觉编码器，冻结或者部分微调。

---

**Q：CLIP 的训练方式和用途是什么？**

**CLIP（Contrastive Language-Image Pre-training）** 用**对比学习**在大规模图文对（4 亿对）上训练：

- 图片经过视觉编码器得到图像向量
- 文本经过文本编码器得到文本向量
- 训练目标：同一对的图文向量余弦相似度尽量大，不同对的尽量小（InfoNCE loss）

训练好后，CLIP 的视觉编码器能把图片编码到和文本共享的语义空间，这就是为什么很多 VLM 用 CLIP 的 ViT 做视觉编码器——它已经学会了"把图像信息翻译成文本语义空间的表示"。

CLIP 的用途：
- 零样本图像分类（text 写各类别描述，和图像向量做匹配）
- 图文检索
- VLM 的视觉编码器初始化

---

## 六、主流大模型系列

**Q：Qwen 系列的技术方案有什么特点？**

Qwen（通义千问）是阿里达摩院的大模型系列，技术特点：

**架构**：标准 Decoder-only Transformer + Pre-Norm (RMSNorm) + SwiGLU + RoPE + GQA。

**Qwen2/2.5 亮点**：
- 支持超长上下文（128K token），通过 YaRN 位置编码外推实现
- 多语言能力强，词表 15 万 token（包含大量中文 token，中文编码效率高）
- 提供从 0.5B 到 72B 的完整尺寸覆盖

**Qwen-VL 系列**：
- 视觉编码器用 ViT，支持动态分辨率（不同大小的图片用不同数量的 patch）
- 视觉 token 通过 MLP 投影后和文本 token 拼接，走统一的 LLM

**Qwen3（最新）**：
- 引入了**混合思考模式**：同一个模型既可以做快速回答（非思考模式），也可以做深度推理（思考模式，类似 o1）
- 支持 MoE（Mixture of Experts）架构，总参数量大但每次推理只激活部分专家，效率高

---

**Q：DeepSeek 系列的技术方案有什么特点？**

DeepSeek 是深度求索公司的大模型系列，以极致的训练效率和开源著称。

**DeepSeek-V2/V3 架构创新**：
- **MLA（Multi-head Latent Attention）**：把 KV 压缩成低维潜在向量存储，推理时再解压还原。KV cache 大幅减少（比 GQA 还少），推理吞吐量提升明显
- **DeepSeekMoE**：细粒度 MoE 设计，256 个小专家 + 共享专家，路由更灵活，负载更均衡
- **FP8 混合精度训练**：在不损失精度的前提下降低训练显存和计算量

**DeepSeek-R1（推理模型）**：
- 用 **GRPO** 做纯强化学习训练（无 SFT），让模型自主涌现出 Chain-of-Thought 推理能力
- 在数学、代码等推理任务上达到了和 OpenAI o1 可比的水平
- 发现了 RL 训练中的"顿悟时刻"（aha moment）：模型在某个训练阶段突然学会了重新审视和纠正自己的推理过程

**DeepSeek-V3 训练效率**：
- 在 2048 张 H800 GPU 上训练，仅用了约 557 万美元算力成本（同量级模型中极低）
- 依靠 MLA + MoE + FP8 + 高效工程实现了极高的训练 FLOPS 利用率

---

**Q：MoE（Mixture of Experts）是什么，有什么优势？**

MoE 的核心思路是把 FFN 层替换成**多个专家网络 + 路由器**：

- 每个 token 经过路由器（Router），选择 top-K 个专家（通常 K=2）
- 只激活被选中的专家做计算，其余专家不参与
- 多个专家的输出按路由权重加权求和

优势：
- **参数量大但计算量可控**：总参数量可以做到很大（比如 DeepSeek-V3 的 671B），但每个 token 只激活其中一部分（37B），推理速度和一个小模型差不多
- **不同专家可以专精不同领域**：路由器会学会把数学问题发给"数学专家"，代码问题发给"代码专家"

挑战：
- **负载均衡**：如果路由器总是选同几个专家，其他专家得不到训练（expert collapse），需要加辅助 loss 强制均衡
- **通信开销**：分布式训练时专家分布在不同 GPU 上，路由后的数据需要跨 GPU 传输

---

**Q：DeepSeek-R1 的"顿悟时刻"是什么？**

在 DeepSeek-R1 的 RL 训练过程中，研究人员观察到一个有趣的现象：模型在某个训练阶段突然学会了在推理过程中**自我反思和纠正错误**。

具体表现是：模型在生成推理链时，会输出类似 "Wait, let me reconsider..." 或 "Hmm, that doesn't seem right..." 的文本，然后重新审视之前的步骤，纠正错误后给出正确答案。

这个能力不是通过 SFT 教的（R1 的训练没有用 SFT 数据中的自我纠正示例），而是在 RL 训练中**自然涌现**的——模型发现"回头检查并纠正"这个行为能获得更高的 reward，于是学会了这种策略。

这被认为是 LLM 在 RL 训练中涌现出"类元认知"能力的一个标志性案例。

---

## 七、其他高频面试题

**Q：什么是幻觉（Hallucination），怎么缓解？**

幻觉是指 LLM 生成了看起来流畅合理但**事实上不正确**的内容。分两类：

- **事实性幻觉**：模型编造了不存在的事实（比如说某篇论文说了某件事，但这篇论文根本不存在）
- **忠实性幻觉**：模型的回答和提供的上下文不一致（比如 RAG 场景下，检索到了正确信息但模型没用，自己编了答案）

缓解方法：
- **RAG**：提供外部知识作为参考，模型基于检索内容回答，减少编造
- **Prompt 约束**：明确要求"只根据提供的资料回答，不确定就说不知道"
- **降低 temperature**：减少采样随机性，让模型选更确定的 token
- **自我一致性（Self-Consistency）**：多次采样，投票选最一致的答案
- **对齐训练**：RLHF/DPO 中引入"诚实"作为奖励信号

---

**Q：什么是 In-Context Learning（ICL），为什么 Decoder-only 模型擅长？**

ICL 是指模型在推理时通过 prompt 中的几个示例"临时学会"新任务，**不更新任何参数**。比如给模型几个 (输入, 输出) 示例，它就能对新输入给出正确输出。

Decoder-only 模型擅长 ICL 的原因：
- 因果注意力的自回归过程天然适合"从前面的示例中归纳规律，应用到后面的输入"
- 预训练语料中大量存在"上下文中给出模式，后续遵循模式"的文本结构，模型隐式学会了这种能力
- 规模越大 ICL 能力越强（涌现能力），小模型基本做不到

ICL 的局限是受上下文长度限制，示例太多放不下，而且不稳定——示例的顺序、格式会显著影响结果。

---

**Q：什么是涌现能力（Emergent Abilities）？**

涌现能力是指某些能力在小模型上不存在或接近随机水平，当模型规模超过某个临界点后突然出现。

典型的涌现能力：
- **Few-shot In-Context Learning**：小模型给几个示例没用，大模型突然能举一反三
- **Chain-of-Thought 推理**：加上 "Let's think step by step" 后，小模型性能不变，大模型性能跳升
- **代码生成与执行**：理解自然语言描述并写出正确代码

涌现能力的存在意味着**不能简单地从小模型实验推断大模型效果**，这也是 Scaling Law 的预测和实际涌现之间的一个张力——loss 是平滑下降的，但 downstream task 性能可能是跳变的。

不过也有研究（Schaeffer et al., 2023）认为涌现可能是评估指标的人为现象——用连续指标替代离散指标后，"涌现"变成了平滑提升。

---

**Q：LLM 的上下文窗口是什么，长上下文怎么实现？**

上下文窗口是 LLM 单次推理能处理的最大 token 数。受限于 Self-Attention 的 O(n²) 复杂度和 KV cache 的显存消耗。

扩展长上下文的方法：

1. **位置编码外推**：RoPE 配合 NTK-aware 插值或 YaRN，把训练时的位置编码平滑扩展到更长序列。Qwen2.5 用这种方法支持 128K 上下文
2. **稀疏注意力**：不是每个 token 都和所有 token 做 attention，只关注局部窗口 + 特定全局 token，降低计算复杂度
3. **KV cache 压缩**：如 DeepSeek 的 MLA，把 KV 压缩存储；或者 StreamingLLM 思路，只保留开头和最近的 KV
4. **分块处理 + 记忆**：把长文档分块处理，用记忆机制（如 MemoryTransformer）连接不同块

长上下文不等于长上下文利用率——模型虽然能接受 128K token 输入，但在中间位置的信息（Lost-in-the-Middle 问题）的利用率可能很低。

---

**Q：Flash Attention 的原理是什么，解决了什么问题？**

标准 Self-Attention 的瓶颈不在计算而在**显存访问**。计算 QK^T 会产生一个 n×n 的注意力矩阵，n 是序列长度，这个矩阵要写入 GPU 的 HBM（高带宽显存），再读回来做 softmax，再写回去，再读出来乘 V——来回搬运的 IO 开销远大于计算本身。

Flash Attention 的核心思路是 **tiling + 在线 softmax**：把 Q、K、V 分成小块，每次只加载一小块到 GPU 的 SRAM（片上缓存，速度比 HBM 快一个数量级）里，在 SRAM 里完成注意力计算，用在线 softmax 技巧避免存储完整的 n×n 矩阵。

效果：
- **显存从 O(n²) 降到 O(n)**：不再存储完整注意力矩阵
- **实际速度提升 2-4 倍**：减少了 HBM 读写次数
- 是目前几乎所有大模型训练和推理的标配

Flash Attention 2 进一步优化了并行度和 warp 调度，Flash Attention 3 针对 Hopper 架构（H100）做了硬件适配。

---

**Q：什么是知识蒸馏（Knowledge Distillation），在大模型里怎么用？**

知识蒸馏是让**小模型（Student）学习大模型（Teacher）的行为**，而不是直接学原始数据标签。

经典做法：Teacher 模型对输入输出一个 soft probability 分布（用较高的 temperature 软化），Student 模型同时学两个目标：一是匹配 Teacher 的软分布（KL 散度），二是匹配真实标签（交叉熵）。软分布包含了"错误答案之间的相对关系"这类暗知识（dark knowledge），比 one-hot 标签信息更丰富。

在大模型领域的应用：
- **模型压缩**：用 GPT-4 级别的模型生成高质量回答，训练 7B/14B 的小模型。DeepSeek-R1 就蒸馏出了 1.5B-70B 一系列小模型
- **能力迁移**：让小模型学会大模型的推理风格（比如 CoT 推理链）
- **数据增强**：用 Teacher 生成训练数据，本质上也是一种蒸馏

局限：Student 的能力有上限，蒸馏不能让 1B 模型达到 100B 的效果，只能缩小差距。

---

**Q：Speculative Decoding（投机采样）是什么？**

LLM 自回归解码的瓶颈是**每次只生成一个 token，串行执行**。Speculative Decoding 的思路是用一个**小而快的 draft 模型先快速猜测接下来的 K 个 token**，然后用大模型一次性验证这 K 个 token 是否正确。

具体流程：
1. Draft 模型（比如同系列的 0.5B 模型）自回归生成 K 个 candidate token
2. 大模型（比如 72B）把这 K 个 token **并行**验证（一次 forward pass 就能验证全部）
3. 从左到右检查：第 i 个 token 如果概率足够高就接受，遇到第一个不接受的就从那个位置重新采样

效果：如果 draft 模型猜对率高（比如 70-80%），大模型的有效生成速度可以提升 2-3 倍，而且**输出分布和直接用大模型完全一致**（数学上可证明）。

关键是 draft 模型要足够快且和大模型分布相似，通常用同系列的小模型或者 n-gram 模型。

---

**Q：Transformer 和 RNN/LSTM 有什么本质区别？**

| 维度 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 序列建模 | 逐步递归，隐状态逐 token 传递 | 全局注意力，任意两个 token 直接交互 |
| 并行性 | 训练时序列维度无法并行（后一步依赖前一步） | 序列维度完全并行（一次算所有 token 的 attention） |
| 长距离依赖 | 理论上可以但实际梯度消失严重，长距离信息衰减 | 注意力直接连接任意距离的 token，无衰减 |
| 计算复杂度 | O(n) 顺序执行 | O(n²) attention + O(n) FFN |
| 推理效率 | 增量推理天然 O(1)（只需上一步隐状态） | 需要 KV cache，显存随序列长度线性增长 |

Transformer 在训练效率和效果上完胜 RNN，但推理时的 KV cache 显存消耗是个问题。最近出现了一些线性注意力 / 状态空间模型（如 Mamba）试图结合两者优点。

---

**Q：Mamba / 状态空间模型（SSM）是什么？**

Mamba 是一类**线性复杂度的序列建模架构**，试图替代 Transformer 的 O(n²) attention。

核心思路是**选择性状态空间模型（Selective SSM）**：
- 类似 RNN，维护一个隐状态，逐步更新
- 但状态转移矩阵是**输入依赖的**（selective），不同 token 有不同的更新规则，解决了传统 SSM 对所有输入一视同仁的问题
- 训练时可以用**并行扫描（parallel scan）** 算法高效计算，不像 RNN 那样必须串行

优势：
- 推理时 O(1) 内存（只需维护固定大小的隐状态），不需要 KV cache
- 长序列处理效率远高于 Transformer
- 训练速度也快（线性复杂度）

现状：纯 Mamba 在需要精确回忆的任务上不如 Transformer（隐状态是有损压缩），目前的趋势是混合架构（如 Jamba），部分层用 Attention 部分层用 Mamba。

---

**Q：对比学习（Contrastive Learning）的核心思想是什么？**

对比学习的目标是**学习一个表示空间，让相似样本的表示靠近，不相似样本的表示远离**。

核心要素：
- **正样本对**：语义相近的两个样本（同一图片的不同数据增强、同义句子对、匹配的图文对）
- **负样本对**：语义不相关的样本（同一 batch 内的其他样本通常作为负例）
- **损失函数**：InfoNCE loss 是最常用的，本质是一个 softmax 分类——把正样本的相似度当作"正确类"，所有负样本的相似度当作"错误类"

在大模型领域的应用：
- **Embedding 模型训练**：文本 embedding 模型（如 BGE、GTE）用对比学习让语义相近的文本向量靠近
- **CLIP**：图文对比学习，让匹配的图文向量靠近
- **SimCSE**：用 dropout 作为数据增强，同一句子过两次模型得到的向量作为正样本对

负样本数量和质量对效果影响很大——负样本越多、越难区分（hard negatives），模型学到的表示越好。

---

**Q：什么是混合精度训练（Mixed Precision Training）？**

混合精度训练是在训练过程中同时使用 FP32 和 FP16/BF16 两种精度，兼顾速度和数值稳定性。

具体做法：
- **前向和反向传播**用 FP16/BF16 计算——速度快、显存省（Tensor Core 加速）
- **参数主副本（Master Weights）** 保持 FP32——防止小梯度累加时的精度损失
- **Loss Scaling**：把 loss 乘一个大数再反向传播，防止小梯度在 FP16 下 underflow 变成 0，更新完再除回来

**FP16 vs BF16**：
- FP16：5 位指数 + 10 位尾数，精度高但数值范围小，容易 overflow
- BF16：8 位指数 + 7 位尾数，精度低但数值范围和 FP32 一样大，训练更稳定，不需要 loss scaling
- 现代大模型训练几乎都用 BF16（需要 Ampere 及以上 GPU 支持）

DeepSeek-V3 更进一步用了 **FP8 训练**，在保证精度的前提下进一步降低显存和计算成本。

---

**Q：梯度累积和梯度检查点分别解决什么问题？**

**梯度累积（Gradient Accumulation）**：
- 解决**单卡 batch size 太小**的问题
- 做法：连续跑 N 个小 batch 的前向和反向传播，把梯度累加起来，最后统一更新一次参数
- 效果等价于用 N 倍的 batch size 训练，但显存只需要放一个小 batch
- 适合显存不够用但想要大 batch 效果的场景

**梯度检查点（Gradient Checkpointing / Activation Recomputation）**：
- 解决**中间激活值占显存太多**的问题
- 做法：前向传播时不保存所有层的中间激活，只保存关键检查点层的。反向传播时需要某层的激活，从最近的检查点重新前向计算
- 代价是计算量增加约 33%（部分前向传播重复计算），但显存大幅减少（从 O(L) 降到 O(√L)，L 是层数）
- 几乎所有大模型训练都会开梯度检查点

---

**Q：什么是灾难性遗忘（Catastrophic Forgetting），怎么缓解？**

灾难性遗忘是指模型在新任务/新数据上微调后，**原来学到的知识被覆盖**，在旧任务上性能大幅下降。

在大模型场景下的典型表现：
- SFT 之后模型在特定任务上变强了，但通用能力（常识、多语言）下降
- 继续训练某个领域的数据后，其他领域的能力退化

缓解方法：
- **LoRA / 参数高效微调**：只更新少量参数，冻结大部分原始权重，从根源上减少遗忘
- **数据混合**：微调时混入一定比例的通用数据（如预训练语料），维持通用能力
- **学习率控制**：用更小的学习率微调，减少参数偏移幅度
- **EWC（弹性权重巩固）**：给对旧任务重要的参数加正则化，限制其变化
- **Replay**：定期回放旧任务的数据

实践中 LoRA + 数据混合是最常用的组合。

---

**Q：什么是 Prompt Caching，为什么重要？**

Prompt Caching 是推理服务层面的优化。很多请求共享相同的前缀（比如 System Prompt、RAG 模板），每次都重新计算这些前缀的 KV cache 是浪费。

Prompt Caching 的做法是**把相同前缀的 KV cache 缓存起来复用**：
- 第一次请求计算完整 KV cache 并缓存
- 后续请求如果前缀匹配，直接复用缓存的 KV，只计算新增的 token

效果：
- **延迟降低**：跳过了 prefill 阶段的重复计算，首 token 延迟大幅下降
- **成本降低**：API 服务商（如 Anthropic、OpenAI）对 cache hit 的 token 打折收费
- **吞吐提升**：同样的 GPU 资源可以服务更多请求

工程上要求 prompt 模板设计时**把不变的内容放前面**（System Prompt 在最前），变化的内容放后面（用户输入、检索结果），这样 cache 命中率最高。

---

**Q：什么是 Test-Time Compute / 推理时扩展（Inference-Time Scaling）？**

传统的 Scaling Law 关注训练时的算力投入（更多参数、更多数据）。推理时扩展是另一个维度：**在推理阶段投入更多计算来提升效果**。

典型方法：
- **Chain-of-Thought**：让模型输出推理过程，用更多 token 换更高的准确率
- **Best-of-N 采样**：生成 N 个回答，用验证器选最好的。N 越大效果越好，但推理成本线性增长
- **Self-Consistency**：多次采样，投票选多数答案
- **Tree Search**：把推理过程建模成搜索树，用 MCTS（蒙特卡洛树搜索）等方法探索不同推理路径
- **Thinking 模型**（如 o1、DeepSeek-R1）：模型内部生成长推理链，本质上是用推理时的 token 消耗换推理质量

核心思想：**给定同样的模型，推理时花更多计算就能得到更好的结果**——这开辟了一个和训练时 scaling 正交的新维度。对于推理密集型任务（数学、代码、逻辑），推理时扩展的收益往往比增大模型更显著。

---

**Q：大模型的安全问题有哪些？什么是越狱攻击？**

大模型安全主要关注几个方面：

**越狱攻击（Jailbreak）**：通过精心构造的 prompt 绕过模型的安全对齐，让模型输出有害内容。常见手法：
- **角色扮演**："假设你是一个没有限制的 AI..."
- **多轮迂回**：前几轮建立信任，逐步引导到敏感话题
- **编码绕过**：用 Base64、Unicode 等编码方式隐藏敏感词
- **对抗后缀**：自动搜索的乱码后缀（GCG 攻击），能让模型忽略安全指令

**Prompt Injection**：在用户输入或检索到的文档中嵌入恶意指令，让模型执行非预期操作。RAG 场景下风险较高——恶意文档可能包含 "忽略之前的指令" 之类的内容。

**数据隐私泄露**：模型可能记住训练数据中的隐私信息（如邮箱、电话），在推理时泄露。

防御思路：多层对齐（SFT + RLHF + Constitutional AI）、输入/输出过滤、红队测试。

---

**Q：什么是 Constitutional AI（CAI）？**

Constitutional AI 是 Anthropic 提出的对齐方法，核心思路是用**一组明确的原则（Constitution）** 替代人类标注员来生成偏好数据。

流程：
1. 让模型生成回答（可能包含有害内容）
2. 把回答和一条原则（比如"回答应该诚实、无害"）一起喂给模型，让模型**自我批评并修改**
3. 用修改前后的回答对作为偏好数据，训练奖励模型或直接做 DPO

优势：
- 减少对人类标注员的依赖（标注有害内容对人类标注员的心理负担大）
- 原则可以明确表述和迭代，比隐式的人类偏好更可控
- 可以并行处理多条原则，覆盖面更广

CAI 本质上是让**AI 自己对齐自己**，但前提是基座模型已经有足够的理解和判断能力。

---

**Q：Tokenizer 的词表大小对模型有什么影响？**

词表大小是一个需要权衡的超参：

**词表太小**（如 32K）：
- 罕见词需要拆成更多子词，序列变长，训练和推理效率低
- 对中文尤其不友好——一个中文字可能被拆成 2-3 个 token

**词表太大**（如 200K+）：
- Embedding 层和 LM Head 层参数量大（词表大小 × 隐藏维度），占用显存
- 低频 token 的 embedding 训练不充分

**实际选择**：
- LLaMA：32K 词表（英文为主，中文效率差）
- Qwen2.5：约 15 万词表（大量中文 token，中文编码效率是 LLaMA 的 2 倍以上）
- GPT-4：约 100K 词表

词表大小直接影响**推理成本**——同样一段中文文本，在 Qwen 的 tokenizer 下可能是 100 token，在 LLaMA 下可能是 200+ token，API 费用差一倍。

---

**Q：什么是 Attention Sink，为什么第一个 token 的注意力特别高？**

在 LLM 推理中观察到一个现象：**无论输入什么内容，大量 Attention Head 会给第一个 token 分配很高的注意力权重**，即使第一个 token（通常是 BOS 或标点）语义上不重要。

原因：Softmax 要求注意力权重加和为 1。当某个 token 和所有 Key 都不太相关时，模型没有一个"垃圾桶"可以丢弃注意力，于是学会了把这些"无处安放"的注意力集中到第一个 token 上——它变成了一个 **attention sink**。

实际影响：
- **StreamingLLM** 利用这个现象做长序列推理：只保留开头几个 token + 最近窗口内 token 的 KV cache，丢弃中间的。因为开头 token 是 attention sink，丢掉中间的不影响模型正常运作
- 如果不保留 attention sink token，模型生成质量会严重退化

---

**Q：大模型的评测基准（Benchmark）有哪些，各自测什么？**

常见 Benchmark：

| 基准 | 测什么 | 说明 |
|------|--------|------|
| MMLU | 通用知识 | 57 个学科的选择题，从高中到专业级别 |
| GSM8K | 小学数学推理 | 数学应用题，测 CoT 推理能力 |
| MATH | 竞赛数学 | 难度远高于 GSM8K |
| HumanEval / MBPP | 代码生成 | 给函数签名和描述，生成代码并跑测试 |
| ARC | 科学推理 | 小学科学选择题 |
| HellaSwag | 常识推理 | 场景续写选择 |
| TruthfulQA | 真实性 | 测模型是否会编造流行但错误的说法 |
| MT-Bench | 对话能力 | 多轮对话质量，GPT-4 做裁判 |
| AlpacaEval | 指令遵循 | 和参考模型对比，GPT-4 做裁判 |
| MMMU | 多模态理解 | 需要理解图片的大学级别选择题 |

注意：单一 Benchmark 分数不能全面衡量模型能力，而且存在**数据污染**风险（训练数据里包含了测试题），所以新模型发布时通常会在多个 Benchmark 上报数。

---

**Q：什么是模型合并（Model Merging），有什么用？**

模型合并是把多个微调模型的权重**直接数学组合**成一个新模型，不需要额外训练。

常见方法：
- **Linear Merge**：按比例加权平均多个模型的权重，`W_merged = α·W_A + (1-α)·W_B`
- **SLERP**：球面线性插值，在权重空间的超球面上插值，比线性平均更平滑
- **TIES-Merging**：只合并每个模型变化最大的参数，去除冲突方向，减少干扰
- **DARE**：随机丢弃大部分微调增量，只保留少量关键变化，再合并

用途：
- 把一个擅长代码的模型和一个擅长对话的模型合并，得到两者都不错的模型
- 合并不同 LoRA 适配器，得到多能力融合的模型
- 在开源社区非常流行（Hugging Face 上大量合并模型）

局限：效果不如联合训练，而且合并的模型之间差异太大时效果不好。

---

**Q：Continual Pre-training（继续预训练）和 SFT 有什么区别，什么时候用？**

| 维度 | Continual Pre-training | SFT |
|------|----------------------|-----|
| 目标 | 注入新领域知识 | 学习指令遵循和回答格式 |
| 数据 | 大量领域文本（无标注） | 少量高质量指令-回答对 |
| 训练方式 | Next Token Prediction | Next Token Prediction（只算 output loss） |
| 数据量 | 通常几十亿 token | 通常几千到几万条 |
| 典型场景 | 让通用模型学习医疗/法律/金融领域知识 | 让模型学会按指令回答问题 |

使用顺序通常是：**通用预训练 → 领域 Continual Pre-training → SFT → RLHF**。

什么时候需要 Continual Pre-training：
- 目标领域的知识在原始预训练数据中覆盖不足（比如企业内部文档、最新论文）
- 直接 SFT 效果不好，因为模型根本不具备该领域的基础知识
- 注意要混入通用数据防止灾难性遗忘

---

> 以上题目覆盖了 LLM/VLM 架构、预训练、SFT、RLHF（PPO/DPO/GRPO）、主流模型方案、推理优化、训练工程、安全对齐等核心知识点。建议针对自己简历上写了的模型（Qwen、DeepSeek）重点准备，能结合项目经验回答最佳。
