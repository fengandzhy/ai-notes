# ai-notes

跟着 [rohitg00/ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
学模型层的笔记。**只记自己动手写出来、或者撞到坑才明白的那部分。**

课程本身不在这个仓库里 —— 它单独 clone 在 `~/workspace/ai-engineering-from-scratch`，
保持只读，这样 `git pull` 拿上游日更永远不会冲突。

和隔壁两个仓库的分工：

| 仓库 | 记什么 |
|---|---|
| [fengandzhy/python](https://github.com/fengandzhy/python) | Python 语法，出处是 langchain demo |
| [fengandzhy/langchain](https://github.com/fengandzhy/langchain) | LangChain / agent 的工程实验 |
| **本仓库** | **模型层**：数学、反向传播、注意力、LLM 训练与推理 |

前两个是「怎么用模型」，这个是「模型里面是什么」。

## 目的

主业是 AI 应用/系统。补这块不是为了转去做建模，是为了**在这条路上不再瞎** ——
当 agent 在某类任务上反复判断错，能说出这是**模型能力边界**还是**提示问题**，
而不是继续往提示词里加字。

## 怎么记

每节一个文件，照抄课程的 `Build It → Use It` 顺序：

1. **先手写**，纯标准库或 NumPy，不许先 import torch
2. **再换库**，看 PyTorch 同样的事怎么做、差在哪
3. 记下**哪一步的直觉是错的** —— 这部分最值钱，课程里没有

每条笔记标出处：`phases/01-math-foundations/01-linear-algebra-intuition`。

## 路线（约 105 小时）

| 阶段 | 节数 | 时长 | 重点 |
|---|---|---|---|
| 01 数学基础 | 22 | 31.8 h | 硬门槛，跳不过去 |
| 03 深度学习核心 | 13 | 18.8 h | `10-mini-framework` 自己写框架 |
| 05 NLP（补 5 节）| 5 | 5.0 h | 07 和 10 的前置，见下 |
| 07 Transformer | 16 | 16.5 h | `02-self-attention-from-scratch` |
| 10 LLM from scratch | 24 | 32.9 h | 终点是 `20-deepseek-v3-walkthrough` |

跳过 02 ML 基础（`03/01` 的前置只要线性代数，实测不是硬依赖）、
04 视觉、06 语音、09 强化学习。

**05 那 5 节是必须补的**，九月定路线时漏了：`07/01` 和 `07/02` 要
`05/09-sequence-to-sequence` + `05/10-attention-mechanism`，`10/01` 要整个 Phase 05。
实际要补：`01-text-processing`、`08-cnns-rnns-for-text`、`09-sequence-to-sequence`、
`10-attention-mechanism`、`19-subword-tokenization`。

节奏按「完成到哪一节」推进，不排日程。

## 环境

Phase 01 前几节**零依赖**，macOS 自带的 `/usr/bin/python3`（3.9.6）直接能跑。
到 Phase 03 才需要 NumPy / PyTorch，那时候再用 uv 建 venv。

## 坑

<!-- 撞到一个记一条 -->
