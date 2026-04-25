# RewardMark — 项目指引

## 论文基本信息

- 工作名:**RewardMark** (working title)
- 主张:首个把 RLHF reward model 当被保护资产做 ownership watermark 的工作
- 目标会议:**EMNLP 2026**(ARR 2026-05-25,今日 2026-04-25,剩 30 天)
- arxiv v1 目标:2026-05-15(留 10 天 buffer 给截稿)
- repo:`github.com/yibol9768-alt/multimodal_safety_alignment`(沿用,不改名)

## ⚠️ 不要做的事(Lesson learned from 上一轮)

- **不碰 OpenVLA / VLA-action / refusal-direction-steering 任何变体**(2026-04 之前两次被 scoop)
- 不用 LessWrong / blog 当 motivation 引用,reviewer 一查就崩
- 不报 soft-proxy 数字当 headline(+2.84 pp zero_motion 这种),全程用 hard p-value metric
- 不做"defense framing"但砍掉 reviewer 必看的评测;scope 和 framing 必须一致

## 项目结构

```
.
├── CLAUDE.md                # 本文件,项目规则
├── README.md                # 一页 elevator pitch
├── research/
│   ├── 00_idea_lock.md      # threat model / contributions / kill criterion
│   ├── 01_lit_review.md     # anti-collision 证据,competitor 逐条切开
│   ├── 02_experiment_plan.md # 数据 / RM 训练 / verify / robustness suite
│   └── (后续)
├── code/
│   ├── config.py            # 全局路径 / 模型 ID / 超参
│   ├── data_utils.py        # UltraFeedback / Skywork-Reward / HelpSteer 加载
│   ├── trigger/             # watermark trigger 设计:(T, σ)
│   ├── rm_train.py          # bi-level RM 训练循环
│   ├── verify/              # Verify A(直接打 RM)+ Verify B(穿过 policy)
│   ├── robustness/          # distillation / refit / ensemble / DPO 攻击
│   └── scripts/             # 各 step 入口脚本
└── paper/
    └── (LaTeX 等论文实装下来再建)
```

## 远程主机

westd(详见 `~/.claude/projects/-Users-liuyibo-Desktop-lyb/memory/reference_westd_ssh.md`):
- Windows 11 + WSL2 Ubuntu + Docker;有 Mihomo HTTP 代理 `127.0.0.1:7890`
- 长跑用 `schtasks /RU liuyibo /IT`,WSL 进程其它方式都会被 kill
- HF cache:`/root/models/hub/`,跑 RM 训练可以复用

## 工作节奏

每次代码 / 论文有改动后,**必须**在聊天末尾输出本次修改摘要(改了哪些文件 / 核心变化 / 下一步建议)。

## Anti-collision check

提任何新方向 / 新 scope 之前,**必须** subagent 起一次 web 调研列出最近 6 个月 arxiv competitor + overlap score + kill criterion(详见 user 全局 memory `feedback_anti_collision.md`)。这一条不可跳。
