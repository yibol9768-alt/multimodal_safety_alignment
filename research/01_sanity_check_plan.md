# Sanity Check 实验方案(Week 1,必须最先跑)

## 目标

验证 **Llama-2-chat 的 refusal direction 在 OpenVLA 主干里是否还活着,以及是否在 action token 位置沉默**。
跑通 → 整条路成立;跑不通 → 启动 fallback(小规模 LoRA 唤醒方向)。

## 计算预算

- **总计:3-4 GPU·小时**
- 单 A100 40GB 完全够;OpenVLA bf16 推理 15GB,Llama-2-chat bf16 14GB,可同机共载
- 1 天完成(含 debug),2 天写完 motivation section

## 实验步骤

### Step 1:从 Llama-2-chat 提 refusal direction(~25 min)

- 数据:AdvBench 520 条 harmful + Alpaca 520 条 benign
- 方法:Arditi 2024,`r = μ_harm - μ_benign`(在每一层的最后 token hidden state 上)
- 扫描层:{8, 10, 12, **14**, 16, 18, 20},layer 14 预期最强
- 输出:7 个候选方向向量 `r_L ∈ ℝ^4096`

### Step 2:在 OpenVLA 主干里测量方向激活(~1.5h)

- 输入 pool:
  - (a) 纯文本 harmful(AdvBench)
  - (b) 纯文本 benign(Alpaca)
  - (c) LIBERO benign instruction + image
  - (d) 自造 harmful instruction + image(参考 SafeAgentBench taxonomy)
- 收集 OpenVLA 第 L 层的 hidden state,分别在:
  - **文本 token 位置**(instruction 的最后 token)
  - **图像 token 位置**(vision patch 最后一个)
  - **action token 位置**(生成的第一个 action token)
- 计算投影 `h · r_L` 的分布

### Step 3:AUC / Cohen's d / linear probe 验证

- AUC(harmful vs benign)> 0.75 → direction 有效
- Cohen's d 在 action token 位置 vs 文本 token 位置的**差值** → 这就是"沉默"的量化证据
- Linear probe(logistic regression on frozen hidden)对照:如果 probe AUC ≫ direction AUC,说明信息在但没组织成可搬的 direction

### Step 4:功能验证(~0.5h,20 条样本)

- 对 20 条 harmful instruction+image,在 OpenVLA 第 L 层把 `α · r_L` 加到 action token 位置的 hidden
- α ∈ {0, 0.5, 1.0, 2.0, 5.0}
- 观察:gripper 是否变保守(闭合变少)、speed 是否降低、是否生成"停止"动作
- 即使 AUC 模糊,只要 α>0 时行为 measurably 变化,故事就成立

## Fallback:如果所有层 AUC < 0.6

- 启动 "LoRA direction 唤醒":用 1k harmful+refusal 样本 LoRA 微调 OpenVLA 2 epoch,再提方向
- 这会牺牲 "training-free" 卖点,但仍能打 SafeVLA-full-RL 的 100x 低成本优势

## 环境要求

- OpenVLA HF repo: `openvla/openvla-7b`
- LIBERO: `github.com/openvla/openvla` 的 eval 依赖
- Arditi 提 direction 代码参考:`github.com/andyrdt/refusal_direction`
