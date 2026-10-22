# ai-notes

跟着 [rohitg00/ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
学模型层的笔记。**只记自己动手写出来、或者撞到坑才明白的那部分。**

课程原文不在这个仓库里 —— 它单独 clone 在 `~/workspace/ai-engineering-from-scratch`，
**保持只读**，这样 `git pull` 拿上游日更永远不会冲突。

这里放的是自己产出的三样东西：**手写代码**、**课文中译**、**错的直觉**。

## 边界

**这个仓库只装一件事：系统学 AI 基础。** 别的都不进来。

### 进这个仓库

- 线性代数、微积分、概率 —— 跟着 Phase 01 手写
- 感知机、反向传播、优化器、自己写的 mini 框架 —— Phase 03
- 注意力、Transformer 各个零件 —— Phase 05 / 07
- tokenizer、预训练、SFT —— Phase 10
- 以及上面这些课文的**中译**（`*_zh.md`）—— 公式改写成纯文本，本地预览器能看

共同点：**都是「模型里面是什么」，而且都得自己写一遍才算数。**

### 不进这个仓库

| 东西 | 去哪 |
|---|---|
| Python 语法、标准库的坑 | [fengandzhy/python](https://github.com/fengandzhy/python) |
| LangChain、agent、工具调用、编排 | [fengandzhy/langchain](https://github.com/fengandzhy/langchain) |
| 提示词技巧、RAG、evaluation、guardrails | 不记 —— 上班天天在做 |
| 部署、网关、灰度、可观测 | 同上 |
| 任何公司项目相关的东西 | 公司仓库，**一行都不要带过来** |

判断标准一句话：**「怎么用模型」不进来，「模型里面是什么」才进来。**

⚠ 最容易破例的是第二行。学到 Transformer 的时候很容易想「顺手把 agent 那套也整理一下」
—— 不要。那是 langchain 仓库的事，混进来这个仓库就废了。

## 目的

补这块不是为了转去做建模，是为了**在这条路上不再瞎** ——
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

## 进度

| 节 | 手写代码 | 课文中译 | 自己挑出来的错直觉 |
|---|---|---|---|
| 01·01 线性代数直觉 | [`01_linear_algebra.py`](phase-01-math/01_linear_algebra.py) | [`01_linear_algebra_zh.md`](phase-01-math/01_linear_algebra_zh.md) | 点积≠相似度 / 形状≠维度 / 相关性不是是非题 / 投影是在扔东西 |

⚠ 中译里的公式**是手写的纯文本图示，不是 LaTeX 渲染**。原文用 ```` ```math ```` 块，
本地预览器大多不认，会把 `\begin{aligned}` 这类排版指令原样印出来。

## 坑

<!-- 撞到一个记一条 -->

- **点积当相似度，只在向量归一化之后成立。** embedding 模型的输出一般是归一化的，
  所以「RAG 用点积」没错；但自己拼过向量（比如把几个 embedding 加起来）就破坏了
  归一化，这时点积会让长的那条永远排第一 —— **不报错，只是结果慢慢变差**
- **`@` 不是 `*`。** NumPy 里 `A * B` 是逐元素乘，`A @ B` 才是矩阵乘。
  形状凑巧兼容时两个都不报错，算出来的东西完全不同
- **所有「等于零」的判断都是「小于某个阈值」。** 同一组向量，`tol=1e-9` 算出秩 2，
  `tol=1e-12` 算出秩 3。真实数据永远是「接近亏秩」，不是干净的满秩或亏秩

## 环境

```bash
uv venv --python 3.14.4     # 建 .venv
uv run python phase-01-math/01_linear_algebra.py
```

PyCharm：解释器指向 `.venv/bin/python`（`设置 → Project → Python Interpreter`）。

**Phase 01 前几节是纯标准库**，不装任何依赖，用 macOS 自带的 `/usr/bin/python3`
（3.9.6）也能跑。建 venv 是为了 PyCharm 的补全和检查，以及后面几节：

```bash
uv add numpy      # 课文 Use It 一节要
uv add torch      # Phase 03 要，包很大，到时候再装
```

⚠ 选 3.14.4 不是随便定的：实测 torch 2.14.1 在 3.12 / 3.13 / 3.14 上都有包，
没有约束，所以用了本机已有的那个（和隔壁 langchain 项目同版本，省一次下载）。
