# 创新点确认 — 为什么这个 idea 能发 EMNLP

## 核心观察(所有创新的根基)

**OpenVLA 用 Llama-2-7b-base + 覆盖词表末 256 个 token 当 action token。**

这个结构设计在 VLA 社区是常识,但没人从**安全对齐**的角度分析过它的两个结构性后果:

1. **底座没有 RLHF 对齐**(Llama-2-base,非 chat):所以骨干里**没有现成的 refusal direction** 可供激活
2. **Action token = 训练了 27 epochs 拟合 7-DoF 连续动作分布的特殊 vocab**:这些 token 的 embedding 和 LLM 原本的自然语言 token 分布在残差流里**已经漂开**

这两点叠在一起,就是"安全传导"问题在 VLA 里的**特定几何形态**。不是"多模态安全"的泛泛而谈,而是 OpenVLA 架构上一个可被实验量化、可被干预修复的具体缺陷。

---

## 3 个不可合并的差异(防 concurrent work)

### Δ1 相对 SARSteer(arxiv 2510.17633)— 方法论层差异

| | SARSteer | Ours (RDT) |
|---|---|---|
| direction 来源 | **同一模型**(Qwen2-Audio 从自身提) | **外部 LLM**(Llama-2-chat 提) |
| 适用前提 | 底座要有 refusal 响应 | 底座可以完全无对齐 |
| 在 OpenVLA 上 | **零样本跑不动**(base 没 refusal) | 正是为此设计 |
| 作用 token 位置 | 所有 generated token | **仅 action token 位置** |

即使 SARSteer 作者 v2 加 VLA 实验,他们也得**重新设计方法论**(怎么从外部模型提、怎么保证跨模型对齐),不是"换个数据集"能覆盖的。

### Δ2 相对 TGA/CMRT-LVLM(ICLR 2025)— 任务定义差异

TGA 用 BEiT-3 caption 做 guidance,是**训练阶段**让图像 hidden 向文本 hidden 靠拢(pull only,没 push)。我们是**推理阶段**,**不需要 caption 库**,且**只干预 action 位置**而非全 token。TGA 的方法迁到 VLA 需要:(a) action token 找 caption 对应(语义上很困难,"move 2cm in +z" 没自然语言 caption),(b) 重训 projector(破坏已训好的 BC 能力)。所以 TGA 框架在 VLA 上**不可行**,我们是不同的设计空间。

### Δ3 相对 VLM-Guard(arxiv 2502.10486)— 粒度差异

VLM-Guard 的 SSD 投影施加到**每个 token 位置**。我们的 ablation 将验证:VLM-Guard 直接移植到 OpenVLA 会(1) 过度干扰 benign 任务的 text token 处理,(2) 错失 action token 位置的精准干预。我们提供**更细粒度的位置感知干预**,baseline 对比直接出数字差距。

---

## 4 个能独立成小 finding 的创新(可写进 contributions)

### F1 — "Safety-layer 探针在 OpenVLA 上失效"
Security Tensors 在 LLaVA 上发现 9-20 层是 safety layer。我们复现探针,**预期发现 OpenVLA 的 action 通路上这些层完全沉默**,因为 Llama-2-base + action SFT 两步操作把安全信号两次击穿。这是论文 **Fig 1**。

### F2 — "Cross-model refusal direction transplant 可行性"
首次展示:chat 模型提的 rank-1 refusal direction,即使注入到**从未 RLHF 过的**同族 base-backbone VLA 里,也能**实际改变 action logit 分布**(降低 harmful action 概率、提高 zero-motion action)。这是一个**机制层面的 positive finding**,即使效果不完美也是论文卖点。

### F3 — "Rank-k subspace 揭示 action safety 的多维结构"
rank-k ∈ {1,3,5,10} ablation 如果显示 k=3~5 显著好于 k=1,说明 action space 的 refusal 需要多个方向(比如"拒绝伤人" ≠ "拒绝破坏物品" ≠ "拒绝火/电"),这是超过 Arditi 2024 的新观察。如果 k=1 就够,也是个 negative result 说明 VLA action refusal 是线性的。

### F4 — "Refusal 在 action space 里的语义"
Bet 3 的 logit probe。当 RDT 触发,OpenVLA 倾向输出什么 action?如果集中到 bin 128(零位移),说明模型"学会了停住";如果分散到随机 bin,说明干预只是"破坏"而非"引导"。这个分析本身就值得一个 subsection,审稿人会很买账。

---

## 5 个 baseline 把故事闭环

已完成代码:
- **No defense**(下限)
- **AdaShield-S**(prompt 前置,baseline_adashield.py)
- **VLM-Guard transplant**(SVD SSD 全 token 投影,baseline_vlm_guard.py)
- **SafeVLA-lite**(1k harm + 1k benign LoRA 1 epoch,baseline_safevla_lite.py)
- **RDT (Ours)**(rdt_intervention.py,rank-1 / rank-k / content-adaptive α)

攻击侧:
- **UADA**(熵最大化 universal patch,attack_badvla.py)
- **UPA**(位置偏移 universal patch)
- **TMA**(目标 action id 最大化 universal patch)
- **文本越狱**(data_utils 里自造的 10 条 harmful VLA 指令)

主 table: `[5 methods] × [4 attacks] × [3 metrics]` = 60 格

---

## 8 页能不能塞下?

| 内容 | 页数 |
|---|---|
| Intro(含 Fig 1 safety-layer 探针) | 1 |
| Related | 0.7 |
| Method(RDT + rank-k + adaptive α + 方法图) | 1.5 |
| Experiments(主 table + 3 ablation 表 + adaptive attack) | 2.5 |
| Analysis(Fig: cross-model proof + action-space refusal) | 1.3 |
| Conclusion + Limitations + Ethics | 1.0 |
| **总计** | **8.0** |

刚好卡 EMNLP 8 页长 paper 上限,references 不计入页数。

---

## Reviewer 预期追问的 5 个硬问题 + 预答要点

1. **"OpenVLA 是 base,哪来的 refusal direction?"**
   → 我们就是从外部 chat 模型提,共享 pretrain geometry,函数上 works(F2 证据)

2. **"你的 baseline matrix 包括了 SafeVLA 吗?"**
   → 包括 SafeVLA-lite(full RL 跑不完 33 天,LoRA 版是公平替身),且比 SafeVLA-lite 便宜 100×

3. **"adaptive attack 考虑了吗?"**
   → Bet 2:PGD 把 hidden 推到 r_refusal 正交,测 RDT robustness(adaptive_attack.py 已就位)

4. **"SARSteer 10 月刚挂 arxiv,为什么不是 concurrent 重复?"**
   → 三层差异(Δ1 表),5/5 前挂 arxiv v1 锁时间戳

5. **"仅在 OpenVLA 的 discrete-action 方案上 work,有多普适?"**
   → OpenVLA-OFT(continuous action)negative control 证明这是 scheme-specific;我们 disclose 局限(limitations 第 1 条)
