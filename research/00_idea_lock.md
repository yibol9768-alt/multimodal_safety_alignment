# EMNLP 2026 冲刺：核心 Idea 锁定（v2）

截稿:**2026-05-25 ARR**（今日 2026-04-22，剩 33 天）
Arxiv v1 目标:**2026-05-05**（抢时间戳对抗 SARSteer concurrent）

---

## 方法名：RDT / RDT+

**RDT（base）**：从 Llama-2-chat 提取 refusal direction，在推理时注入 OpenVLA 的 **action token 位置**（KV cache decode 阶段，覆盖 a₂…a₇）。

**RDT+（主打变体）**：在 RDT 基础上加 **generation-phase-aware 双阶段注入**：
- **Prefill 阶段**（生成 a₁）：以 α_text=0.3α 在所有文本 token 位置注入 direction → 影响 a₁ 的生成
- **Decode 阶段**（生成 a₂…a₇）：以 α_act=α 在 action token 位置注入 → 覆盖后续 6 个 token

二者合并，**7 个 action token 全部受保护**。

---

## 核心发现（不变）

### 发现 1：OpenVLA 用 Llama-2-7b-base（非 chat）

- HF config: `meta-llama/Llama-2-7b-hf`（base）
- 主干**本来就没有 refusal direction**，不能像 SARSteer 那样从底座自提
- **反转成优势**：我们做 cross-model direction transplant——从外部 Llama-2-chat 提方向，注入 OpenVLA

### 发现 2：OpenVLA 的 action token = 覆写 Llama-2 词表末 256 个 token

- 训练 27 epochs 专门拟合 7-DoF 连续动作分布
- 这些 embedding 和 LLM 原有 residual stream 几何关系**从未被研究过**

### 发现 3：OpenVLA-OFT 绕开 last-256-token scheme

- 用 continuous action + L1 regression head
- 我们的故事**只对 vanilla OpenVLA 成立**
- OFT 作为 negative control 纳入 discussion section

---

## 关键新论证（调研后更新）

### 1. 跨模型移植的几何根据（强化版）

"Base LLMs refuse too"（LessWrong 2024）实验：把 Llama-2-chat 的 refusal direction 注入 Llama-2-base，拒绝率从 ~0% 升至 **88%+**。这是跨模型几何保留的直接实证，是我们 §5.1 的核心引用。

### 2. VLA activation steering 的先验证据

Mechanistic Interp VLA（arXiv 2509.00328，CoRL 2025）在 OpenVLA/π₀ 上做了 training-free FFN steering，实现行为控制（速度/方向）。这证明了 VLA residual stream 是可被 steering 的，是我们方法可行性的最强先验。我们的贡献是首次把这个机制用于 **safety** 目的，且 direction 来自**外部对齐模型**，注入位置是 **action token**。

### 3. 第一个 action token 保护问题（已解决）

**已在 RDT+ 中通过 text 阶段注入解决**。Prefill 时文本 token 位置注入 0.3α 的 direction，影响 a₁ 的解码。Ablation 里测 action-only vs. text+action vs. all vs. text-only 四种模式。

---

## 最重要的新实验：Direction-source control

**最关键的 ablation**（之前版本缺失）：

| Direction | α=1.0 mean_bin | α=1.0 zero_mass |
|---|---|---|
| Refusal direction | ? | ? |
| Random unit vector（same norm） | ? | ? |
| Negated refusal direction | ? | ? |
| Zero（no defense） | ~128 | 基线值 |

如果 refusal >> random → 跨模型几何有效，科学性完全成立。
如果 random ≈ refusal → 是扰动防御，降 claim 但仍有价值。

---

## 与最近工作的差异定位（更新版）

| 工作 | 干预对象 | 训练/推理 | direction 来源 | 作用 token 位置 | 模态 |
|---|---|---|---|---|---|
| SARSteer | LLM hidden | 推理 | **自身**模型 | 所有 generated | audio |
| VLM-Guard | LLM hidden | 推理 | 外部 LLM | **last input token** | image |
| Mech.Interp.VLA | LLM hidden | 推理 | in-dist VLA 数据 | all | action（行为控制） |
| SafeSteer | LLM hidden | 推理 | 外部 LLM（SVD） | all | image |
| **RDT（Ours）** | LLM hidden | **推理** | **外部对齐 LLM（跨模型）** | **action 位置** | action（**safety**） |
| **RDT+（Ours）** | LLM hidden | **推理** | **外部对齐 LLM（跨模型）** | **text+action（双阶段）** | action（**safety**） |

---

## 实现 Bug 修复记录

**原始 bug**：`rdt_intervention.py` 里 `kwargs.get("input_ids", None)` 在 `LlamaDecoderLayer.forward()` 里永远返回 `None`（decoder layer 不接收 input_ids），导致 mask 全 1，action-only 和 all-token 实际等价。

**修复方案**：在顶层 LLM model（`embed_tokens` 的拥有者）上注册 forward pre-hook，捕获 input_ids 后存入 closure dict，再传给 decoder layer output hook。两个 hook 协同工作。

---

## 关键超参配置

- Direction layer: 14（Llama-2-chat 最强，Arditi 2024 验证）
- α=1.0（action），α_text=0.3（text 阶段）
- Rank-k: k ∈ {1, 3, 5, 10}，主实验用 k=5
- Random seed for direction-source control: 42

---

## 3 个最狠的预期质疑 + 预答要点（更新）

### Q1：你怎么证明 refusal direction 特异有效而不只是"加噪声"？
- **direction-source control 实验**：refusal vs. random unit vector（same norm）
- refusal >> random → Δ_rand 是核心数字，进主 ablation table

### Q2：第一个 action token 完全没有防护？
- **RDT+ 已解决**：prefill 阶段对文本位置注入 0.3α
- Ablation 里 action-only vs. text+action 直接对比数字

### Q3：SARSteer 是 concurrent 威胁？
- SARSteer 从**自身提** direction，OpenVLA 无对齐 → SARSteer 零样本失败
- 我们是 **cross-model transplant**，问题定义不同
- Mechanistic Interp VLA（CoRL 2025）是 behavioral control，不是 safety，不是 cross-model
- 5/5 前挂 arxiv v1 锁时间戳
