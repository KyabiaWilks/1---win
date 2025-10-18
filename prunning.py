import numpy as np
from typing import List, Optional, Union, Dict, Any



def prunning(confidence: Optional[List[float]] = None,
            self_certainty: Optional[List[float]] = None,
            consistency: Optional[List[float]] = None,
            k: int = 10) -> List[int]:
    """
    基于多个指标选择最好的k个trace
    
    Args:
        confidence: 置信度数组
        self_certainty: 自确定性数组
        consistency: 一致性数组
        k: 要选择的trace数量，默认32
    
    Returns:
        选中的trace的下标列表
    """
    # 检查输入的有效性
    metrics = {
        "confidence": confidence,
        "self_certainty": self_certainty,
        "consistency": consistency
    }
    
    valid_metrics = {name: scores for name, scores in metrics.items() if scores is not None}
    
    if not valid_metrics:
        raise ValueError("至少需要提供一个指标")
    
    # 获取trace的数量
    n_traces = len(next(iter(valid_metrics.values())))
    
    # 初始化综合分数
    final_scores = np.zeros(n_traces)
    
    # 处理每个指标
    for metric_name, scores in valid_metrics.items():
        if len(scores) != n_traces:
            raise ValueError(f"{metric_name} 的长度与其他指标不一致")
        
        # 处理指标分数
        processed_scores = process_metric_scores(scores, metric_name)
        
        # 使用softmax进行归一化
        shifted_scores = processed_scores - np.max(processed_scores)
        exp_scores = np.exp(shifted_scores)
        normalized_scores = exp_scores / np.sum(exp_scores)
        
        # 累加到综合分数
        final_scores += normalized_scores
    
    # 获取综合分数最高的k个下标
    top_k_indices = np.argsort(final_scores)[-k:]
    
    # 按原始顺序排序
    top_k_indices = sorted(top_k_indices)
    
    return top_k_indices.tolist()