import json
import re
from collections import defaultdict
import numpy as np
from typing import List, Optional, Dict, Any, Union

if __name__ == "__main__":
    # 测试所有可用指标
    metrics_to_test = [
        "confidence",
        "TEMP",
        # "self_certainty",
        # "consistency"
    ]
    
    # 测试不同的权重指数
    powers_to_test = [0.0, 0.5, 1.0, 2.0]
# 宏定义 (全局配置)
# ==============================================================================
# 权重指标选择
WEIGHT_METRIC = "confidence"  # 可选: "confidence", "self_certainty", "semantic_entropy", "consistency"

# 置信度指数
CONFIDENCE_POWER = 0.5

# 处理的trace数量
MAX_TRACES = 64


# ==============================================================================
# 核心函数
# ==============================================================================


def check_answer_equality(pred: str, correct: Union[str, int]) -> bool:
    """检查预测答案和正确答案是否等价"""
    if not pred or correct is None:
        return False
    
    # 将correct_answer转换为整数（如果是字符串形式的数字）
    if isinstance(correct, str) and correct.isdigit():
        correct = int(correct)
    
    # 如果correct_answer是数字，将字母转换为对应数字
    if isinstance(correct, int):
        if pred in 'ABCD':
            return ord(pred) - ord('A') == correct
        return False
    
    # 如果correct_answer是字母，直接比较
    return pred == correct

def get_value_from_arrays(item: Dict[str, Any], metric_name: str, index: int) -> Optional[float]:
    """
    从指标数组中获取对应位置的值
    
    Args:
        item: 数据项
        metric_name: 指标名称
        index: 数组索引
    
    Returns:
        Optional[float]: 指标值，如果不存在则返回None
    """
    values = item.get(metric_name)
    if not values or not isinstance(values, list) or index >= len(values):
        return None
    return values[index]

def normalize_metric(value: float, metric_name: str) -> float:
    """
    标准化指标值
    
    Args:
        value: 原始指标值
        metric_name: 指标名称
    
    Returns:
        float: 标准化后的值（越大越好）
    """
    if metric_name == "TEMP":  # TEMP越小越好，使用1/(1+x)转换
        return 1.0 / (1.0 + value)  # 使得温度越小权重越大
    elif metric_name == "self_certainty":  # self_certainty越大越好，但可能为负
        return np.exp(value)  # 使用指数转换，保证正值
    return value  # confidence和consistency默认越大越好

def weighted_vote_from_json(json_file_path: str, 
                      metric_name: str = WEIGHT_METRIC,
                      confidence_power: float = CONFIDENCE_POWER) -> List[str]:
    """
    处理JSON文件中的多个问题，每个问题有多个trace
    
    Args:
        json_file_path: JSON文件路径
        metric_name: 使用的指标名称
        confidence_power: 权重指数，用于控制指标值的影响程度
    
    Returns:
        List[str]: 每个问题的最终答案列表
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取文件 {json_file_path}: {str(e)}")
        return []

    final_answers = []
    
    # print(f"\n开始处理文件: {json_file_path}")
    # print(f"使用指标: {metric_name}")
    # print(f"权重指数: {confidence_power}")
    # print("-" * 50)

    for q_idx, question in enumerate(questions):
        #print(f"\n处理问题 {q_idx + 1}/{len(questions)} (ID: {question.get('id', 'unknown')})")
        
        outputs = question.get("output", [])
        if not outputs:
            #print(f"警告: 问题 {q_idx + 1} 没有trace数据")
            final_answers.append(None)
            continue

        scores = defaultdict(float)
        valid_traces = 0
        
        # 处理每个trace
        for i, output in enumerate(outputs[:MAX_TRACES]):
            # 获取答案（从output或answer数组）
            answer = question.get("answer", [])[i]
            metric_value = get_value_from_arrays(question, metric_name, i)
            
            # 检查答案是否为有效的ABCD
            if not answer or answer not in 'ABCD' or metric_value is None:
                #print(f"Trace {i+1}: 跳过 (答案无效或{metric_name}指标缺失)")
                continue
            
            # 计算权重并累加分数
            weight = normalize_metric(metric_value, metric_name) ** confidence_power
            scores[answer] += weight
            valid_traces += 1
            
            #print(f"Trace {i+1}: 答案={answer}, {metric_name}={metric_value:.4f}, 权重={weight:.4f}")

        if not scores:
            #print(f"问题 {q_idx + 1}: 没有有效的答案")
            final_answers.append(None)
            continue

        # 选择得分最高的答案
        final_answer = max(scores.items(), key=lambda x: x[1])[0]
        final_answers.append(final_answer)
        
        # 输出结果统计
        correct_answer = question.get("correct_answer")
        is_correct = check_answer_equality(final_answer, correct_answer)
        

    return final_answers

def calculate_accuracy(predictions: List[str], json_data: List[dict]) -> float:
    """
    计算预测结果的准确率
    
    Args:
        predictions: 预测的答案列表
        json_data: 原始数据，包含正确答案
    
    Returns:
        float: 准确率（0-1之间）
    """
    if len(predictions) != len(json_data):
        raise ValueError("预测答案和原始数据数量不匹配")
    
    correct_count = 0
    total_count = 0
    
    for pred, question in zip(predictions, json_data):
        correct_answer = question.get("correct_answer")
        if correct_answer is not None:  # 只统计有正确答案的题目
            total_count += 1
            if check_answer_equality(pred, correct_answer):
                correct_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0

def run_weighted_voting(json_file_path: str, metric: str, power: float) -> Dict[str, Any]:
    """
    运行加权投票，使用给定的单个指标和权重，返回结果
    
    Args:
        json_file_path: 输入JSON文件路径
        metric: 要使用的指标，如 "confidence", "TEMP", "self_certainty"
        power: 要使用的权重指数，如 0.5, 1.0

    Returns:
        Dict[str, Any]: 结果字典，格式为：
        {
            'results': List[str],  # 预测结果
            'accuracy': float,     # 准确率
            'metric': str,         # 使用的指标
            'power': float         # 使用的权重指数
        }
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"错误：无法读取文件 {json_file_path}: {str(e)}")
        return {}
    
    # 运行投票
    predictions = weighted_vote_from_json(
        json_file_path,
        metric_name=metric,
        confidence_power=power
    )
    
    # 计算准确率
    accuracy = calculate_accuracy(predictions, json_data)
    
    
    # 返回结果
    return accuracy

def save_results(results: List[str], json_data: List[dict], metric_name: str, confidence_power: float):
    """
    保存结果到文件，包含正确答案信息
    
    Args:
        results: 预测的答案列表
        json_data: 原始问题数据
        metric_name: 使用的指标名称
        confidence_power: 使用的权重指数
    """
    # 构建包含预测答案和正确答案的结果列表
    detailed_results = []
    for i, (pred_answer, question) in enumerate(zip(results, json_data)):
        detailed_results.append({
            "id": question.get("id", str(i)),
            "predicted": pred_answer,
            "correct_answer": question.get("correct_answer"),
            "is_correct": check_answer_equality(pred_answer, question.get("correct_answer")) if pred_answer else None
        })

    # 构建输出文件名
    output_file = f"output_{metric_name}_power_{confidence_power:.1f}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metric_used": metric_name,
            "confidence_power": confidence_power,
            "max_traces": MAX_TRACES,
            "results": detailed_results
        }, f, indent=4)

if __name__ == "__main__":
    # 测试样例
    result = run_weighted_voting(
        "result/task1",
        metric="confidence",
        power=0.5
    )
    
    # 打印结果
    print(f"\n指标: {result['metric']}")
    print(f"权重指数: {result['power']}")
    print(f"准确率: {result['accuracy']:.2%}")