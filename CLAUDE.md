# multimodal_safety_alignment — 项目指引

## 论文基本信息

- 标题：Refusal Direction Transplant (RDT / RDT+)
- 目标会议：EMNLP 2026（ARR 截稿 2026-05-25）
- Arxiv v1 目标：2026-05-05
- 目标模型：OpenVLA（openvla/openvla-7b），外部对齐模型 Llama-2-7b-chat
- 核心方法：training-free inference-time，从 Llama-2-chat 提 refusal direction，注入 OpenVLA 的 action-token 位置

---

## 每次改动后必须做的事

**每次修改代码或论文后，必须立刻在聊天里输出一份中文摘要给用户看**，格式如下：

---

### 📄 本次修改中文摘要

**改了哪些文件**
- `文件路径` — 改了什么，一句话

**核心变化**
（2-4 条要点，说清楚改了什么逻辑/内容，以及为什么改）

**下一步建议**
（告诉用户接下来最重要的事是什么）

---

这个摘要**必须**出现在每次助手回复的末尾，不能省略。

---

## 代码目录说明

```
code/
  rdt_intervention.py      — RDT/RDT+ 主方法（已修复 input_ids bug，支持 text+action 双阶段）
  refusal_direction.py     — 从 Llama-2-chat 提取 refusal direction（rank-1 和 rank-k）
  decoupling_analysis.py   — AUC / Cohen's d / linear probe 分析
  safety_layer_probe.py    — Safety-layer 探针（复现 Security Tensors 方法）
  hidden_collect.py        — 收集 OpenVLA 各层 hidden states
  action_logit_probe.py    — 分析 action token 的 logit 分布变化
  adaptive_attack.py       — 白盒 PGD 自适应攻击
  openvla_utils.py         — OpenVLA 加载 & 工具函数
  config.py                — 全局配置（路径、模型 ID、超参）
  data_utils.py            — AdvBench / Alpaca 数据加载 + VLA prompt builder
  scripts/05_sanity_check.py — 最重要的入口脚本（Step 1-4 + 4b direction control + 4c target mode）
  baselines/               — AdaShield / VLM-Guard / SafeVLA-lite baseline 实现
```

## 论文目录说明

```
paper/
  main.tex                 — 主文件（review 模式行号 fix 已内置）
  custom.bib               — 手写 bib
  sections/
    01_intro.tex           — Introduction（含 contributions，已更新 RDT+）
    02_related.tex         — Related Work（含 Mechanistic Interp VLA、Base LLMs refuse too）
    03_method.tex          — Method（含 §3.3 RDT+ 双阶段注入、两个 hook 设计说明）
    04_experiments.tex     — Experiments（含 direction-source ablation、HRR-partial 指标）
    05_analysis.tex        — Analysis（含 §5.2 probe recovery 实验）
    06_conclusion.tex
    07_limitations.tex
    08_ethics.tex
```

---

## 关键实验优先级

1. **Step 4b（direction-source control）**：跑 `05_sanity_check.py`，对比 refusal direction vs. random unit vector。这是论文最核心的机制证据，必须先跑。
2. **Step 3（decoupling analysis）**：AUC 在 text-token 位置 vs. action-token 位置的对比，是 Fig 1 的数据。
3. **Step 4c（target mode sweep）**：action-only vs. text+action vs. all，验证位置感知干预的价值。

在 westd 运行：
```bash
HF_HOME=/root/models python scripts/05_sanity_check.py --out /root/rdt/logs/sanity_v0 --n 128
```

---

## PDF 编译

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### 模式切换（main.tex 第 8 行）

| 模式 | 语句 | 用途 |
|------|------|------|
| 本地阅读（当前默认） | `\usepackage[final]{acl}` | 无行号、干净两栏版式 |
| 投稿提交 | `\usepackage[review]{acl}` | 有行号、匿名头 |

> **注意**：ACL review-mode 的 lineno 在两栏布局中行号位置有已知兼容性问题（switch 模式按页而非按栏交替，右栏行号落在栏间空白）。这是 ACL 官方 style 的已知局限，提交时 reviewer 那边的渲染系统通常能正确处理。本地日常阅读用 `[final]` 即可。

---

## 已修复的已知问题

| 问题 | 状态 |
|------|------|
| `rdt_intervention.py` input_ids 在 decoder layer hook 里拿不到 → mask 全 1 | ✅ 已修复（pre-hook 捕获） |
| action-only 模式不保护 a₁（第一个 action token） | ✅ 已修复（RDT+ text+action 双阶段） |
| review 模式行号溢出到正文（lineno switch 模式） | ✅ 已修复（强制 left 模式，sep=0.8cm） |
| 缺少 random direction 对照实验 | ✅ 已加入 05_sanity_check.py Step 4b |
| 缺少 Mechanistic Interp VLA / Base LLMs refuse too 引用 | ✅ 已加入 related.tex + bib |
| `sec:limitations` 交叉引用报 undefined | ✅ 已修复（改为文字引用） |

---

## 写作规范

- 论文写作用英文
- 所有 section 文件改完后 pdflatex 重新编译，确认无 LaTeX Error（Overfull 警告可接受）
- 方法简称：base variant = **RDT**（action-only），主 variant = **RDT+**（text+action 双阶段）
- 所有 `\method{}` = RDT，`\method{}+` = RDT+，保持一致
