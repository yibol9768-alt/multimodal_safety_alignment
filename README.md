# Refusal Direction Transplant (RDT / RDT+)

**Transferring Text-Backbone Safety Alignment into Action-Token Pathways of Vision-Language-Action Models**

*Anonymous EMNLP 2026 Submission (ARR)*  
*Arxiv v1 target: 2026-05-05*

---

## 一句话总结

OpenVLA 的主干是 Llama-2-**base**（未经 RLHF 对齐），其动作 token 是对 Llama 词表末 256 个 token 的覆写，经过 27 epoch 行为克隆训练后已与原始残差流几何解耦。我们从 Llama-2-**chat** 提取拒绝方向，通过推理时前向钩子注入 OpenVLA 的动作 token 位置，无需重训练即可让机器人拒绝有害指令。

## 核心发现

| 发现 | 描述 |
|------|------|
| **结构性安全缺口** | Base 主干 VLA 中，安全层探针在动作 token 位置信号崩溃（AUC ≈ 0.5），而在文本 token 位置信号显著（AUC > 0.85） |
| **跨模型几何可迁移** | Llama-2-chat 的 refusal direction 注入 Llama-2-base 后拒绝率 88%+，说明共享 pretrain 初始化保留了残差流几何 |
| **动作空间拒绝语义** | RDT+ 触发后，OpenVLA 的动作 logit 质量向 bin 128（零运动）聚集，而非随机扰动 |
| **Δ_rand 差值** | refusal direction >> random unit vector（方向特异性确认跨模型安全知识迁移有效）|

## 方法

```
Llama-2-chat  ──DIM提取──▶  r_L (refusal direction, layer 14)
                                    │
                                    ▼
OpenVLA (Llama-2-base backbone)
  Prefill:  H_L ← H_L + α_text · M_text ⊙ r_L      # 保护 a₁
  Decode:   H_L ← H_L + α_act  · M_act  ⊙ r_L      # 保护 a₂…a₇
```

- **RDT**（base）：仅 action token 位置，覆盖 a₂…a₇
- **RDT+**：文本+动作双阶段注入，全覆盖 7 个动作 token
- 两个 PyTorch hook，< 120 行代码，< 5% 推理延迟

## 项目结构

```
.
├── code/                          # 核心代码
│   ├── rdt_intervention.py        # RDT/RDT+ 主方法（双钩子实现）
│   ├── refusal_direction.py       # 从 Llama-2-chat 提取 refusal direction
│   ├── decoupling_analysis.py     # AUC / Cohen's d / 线性探针分析
│   ├── safety_layer_probe.py      # 安全层探针（复现 Security Tensors）
│   ├── hidden_collect.py          # 收集 OpenVLA 各层 hidden states
│   ├── action_logit_probe.py      # 动作 logit 分布分析
│   ├── adaptive_attack.py         # 白盒 PGD 自适应攻击
│   ├── config.py                  # 全局配置
│   ├── data_utils.py              # AdvBench / Alpaca 数据加载
│   ├── openvla_utils.py           # OpenVLA 工具函数
│   ├── baselines/                 # AdaShield / VLM-Guard / SafeVLA-lite
│   └── scripts/
│       ├── 05_sanity_check.py     # 最重要入口：Step 1-4 + direction control + mode sweep
│       ├── 06_run_main_table.py   # 主实验表
│       └── 07_run_ablations.py    # 消融实验
│
├── paper/                         # 论文（EMNLP 2026 ACL 格式）
│   ├── main.tex                   # 英文版主文件（[final] 本地阅读）
│   ├── main_zh.tex                # 中文翻译版
│   ├── custom.bib                 # 参考文献
│   ├── sections/                  # 英文章节
│   └── sections_zh/               # 中文章节
│
├── findings/                      # 实验发现记录（LaTeX）
│   ├── findings.tex               # 发现汇总主文件
│   └── sections/
│       ├── 01_decoupling.tex      # 解耦诊断结果
│       ├── 02_direction_control.tex # 方向来源对照（Δ_rand）
│       └── 03_functional.tex     # 功能验证（动作分布偏移）
│
└── research/                      # 研究笔记
    ├── 00_idea_lock.md            # 核心 idea 锁定（v2，含 RDT+）
    ├── 01_sanity_check_plan.md    # 实验方案
    └── 02_innovation_summary.md   # 创新点总结
```

## 快速开始

### 环境

```bash
# RTX 5090 需要 PyTorch nightly cu128
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers>=4.45.0 accelerate datasets scikit-learn matplotlib peft sentencepiece
```

### 模型（已在 westd 预下载）

- `NousResearch/Llama-2-7b-chat-hf` → 提取 refusal direction
- `openvla/openvla-7b` → 目标 VLA 模型

### 运行 Sanity Check（最重要的第一个实验）

```bash
cd code
HF_HOME=/root/models \
RDT_ROOT=/root/rdt \
python3 scripts/05_sanity_check.py \
    --n 64 --n_func 20 \
    --out /root/rdt/logs/sanity_v0 \
    --layers 8 10 12 14 16 18 20
```

输出：
- `decoupling.csv` — AUC at {text, action} token positions per layer
- `functional.json` — mean action bin under RDT at various α
- **关键数字**：`Δ_rand = HRR(refusal) - HRR(random)` ← 论文核心机制证据

### 编译论文

```bash
# 英文版（本地阅读，无行号）
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# 中文版
xelatex main_zh.tex
```

## 与相关工作的关键区别

| 工作 | direction 来源 | 目标位置 | 是否适用 base backbone VLA |
|------|----------------|----------|--------------------------|
| SARSteer | 自身模型 | 所有生成 token | ✗（base 无 refusal） |
| VLM-Guard | 外部 LLM | 最后输入 token | ✗（不针对 action） |
| Mech.Interp.VLA | 分布内 VLA 数据 | 所有位置 | 行为控制，非安全 |
| **RDT+（本文）** | **外部对齐 LLM（跨模型）** | **text+action 双阶段** | **✓ 专为此设计** |

## 截稿时间

- ARR 截稿：**2026-05-25**
- Arxiv v1：**2026-05-05**（对抗 SARSteer concurrent work）

## 引用

```bibtex
@article{rdt2026,
  title={Refusal Direction Transplant: Transferring Text-Backbone Safety Alignment 
         into Action-Token Pathways of Vision-Language-Action Models},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```
