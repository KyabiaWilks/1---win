# ZeroEval

## 项目结构

### 核心文件
- `numerical_calculator.py` - 主程序：两阶段数值计算与投票流程
- `extension_calculator.py` - 扩展计算器：语义不确定性等高级指标
- `prunning.py` - 基于前缀质量的剪枝算法
- `voting.py` - 加权投票算法
- `output_structure_definition.py` - 输出数据结构定义

### 工具文件夹
- `parameter_search_tools/` - 参数搜索工具集
- `dataset_filtering_tools/` - 数据集筛选工具
- `experimental_pkl_logits/` - 实验性PKL/logits提取工具
- `src/` - 源代码模块
- `test_data/` - 测试数据

### 快速开始

```bash
# 运行核心评估流程（两阶段计算+剪枝+投票）
python numerical_calculator.py \
    --input_file test_data/mmlu-redux/openPangu-Embedded-7B_n64.json \
    --max_items 50 \
    --run_prunning_voting \
    --k 10 \
    --first_stage_token_limit 10 \
    --voting_power 0.5 \
    --voting_metric confidence
```

---

# 核心算法
## Metrics
题目中引用的平均置信度评估方法存在一个显著缺陷：它平等地对待推理路径中的每一个词元（token）。然而，一个冗长但最终错误的推理，可能因包含大量高置信度的模板化短语（如‘First, the question is’等）或简单的中间步骤，而被错误地赋予高分。为了解决这个问题，我们提出了一种关键节点加权策略。我们假设，真正的逻辑推理发生在逻辑连接词（如because等）附近。因此，我们识别这些关键逻辑词，并显著提升其邻近词元的权重，从而使我们的评估模型能更精准地聚焦于推理链中的核心逻辑环节，而非无差别的文本流畅度。

具体指标：

- 语义熵
- self certainty
- confidence
- semantic uncertainty


## Voting
参考题目给出算法，加权投票。首先对每个结果的置信度分数进行标准化处理。随后，我们引入一个超参数——权重指数 p，对标准化后的置信度进行幂次变换。这个指数 p 作为一个置信度校准器，允许我们灵活地调整高置信度结果的影响力。最后，我们将所有指向同一答案选项的权重进行累加，形成该选项的最终得分。得分最高的选项即为我们最终采纳的答案。

$$
S(a_j)=\Sigma_{i=1}^N (Norm(m_i))^p\times \mathbb{I} (Ans(t_i)=a_j)
$$

最终选择 $Ans_{final}=arg max_{a_j \in A}S(a_j)$

## Prunning
为了提升大规模生成任务的计算效率与最终输出质量，我们引入了一种基于前缀质量预测的启发式剪枝策略。我们的基本假设是：一个高质量生成序列的初始部分（即前 k 个词元），其内在质量指标也同样会表现优异。算法流程如下：

1. 局部特征提取: 对每个trace，我们仅在其初始的 k 个词元上计算一组异构的metrics
2. 跨路径分数校准: 为消除不同指标的量纲并引入竞争性排序，我们对每个指标在所有 traces 上的得分应用 Softmax 函数进行归一化。这使得每个 trace 在单一指标上的得分，反映了其在该维度上的相对优势。
3. 单路径分数聚合: 我们将每个 trace 经过归一化后的所有指标分数进行线性求和，形成一个能够表征其早期综合质量的最终分数。
4. Top-m 筛选: 基于这个综合分数，我们仅保留排名最高的 m 个 traces，并剪除其余。

本质上是一个低成本的代理模型，它用生成初期的可计算特征，来预测并筛选具有高潜力完成质量的序列，实现了在推理效率和生成质量之间的有效平衡。

$$
S_i = \sum_{j=1}^{P} \frac{e^{f_j(s_{ij})}}{\sum_{k=1}^{N} e^{f_j(s_{kj})}}
$$

最终选择：$I_{top\_k} = \text{argsort}(S_1, S_2, ..., S_N)[-k:]$



# 参考文献

1. Farquhar, Sebastian, et al. "Detecting Hallucinations in Large Language Models Using Semantic Entropy." *Nature*, vol. 630, 2024, pp. 625-30, https://doi.org/10.1038/s41586-024-07421-0.

2. Holtzman, Ari, et al. "The Curious Case of Neural Text Degeneration." *International Conference on Learning Representations*, 2020. *arXiv*, https://arxiv.org/abs/1904.09751.

3. Kang, Zhewei, et al. *Scalable Best-of-N Selection for Large Language Models via Self-Certainty*. arXiv, 2025, https://arxiv.org/abs/2502.18581.

4. Lightman, Hunter, et al. "Let's Verify Step by Step." *International Conference on Learning Representations*, 2024. *arXiv*, https://arxiv.org/abs/2305.20050.

5. Lippi, Marco, and Paolo Torroni. "Argumentation Mining: State of the Art and Emerging Trends." *ACM Transactions on Internet Technology*, vol. 16, no. 2, 2016, pp. 1-25, https://doi.org/10.1145/2850417.