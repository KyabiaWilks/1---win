#!/usr/bin/env python3
"""
参数搜索脚本 - 并行搜索最优参数组合
遍历五个参数：voting_power, first_stage_token_limit, k, keyword_weight_multiplier, keyword_window_size
记录准确率、token总数等结果
"""

import json
import subprocess
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import threading
import time

# ==================== 控制宏 ====================

# 控制是否参与搜索的宏（True=固定值，False=参与搜索）
FIXED_MAX_ITEMS = True  # True: 固定1000题; False: 参与循环
FIXED_TOKEN_LIMIT = True # True: 固定为0; False: 参与循环(20-100)

# ==================== 参数搜索空间定义 ====================

def generate_range(start, end, step):
    """生成浮点数范围（包含end）"""
    result = []
    current = start
    while current <= end + 1e-9:  # 加一个小epsilon避免浮点误差
        result.append(round(current, 2))
        current += step
    return result

PARAMETER_GRID = {
    "voting_power": generate_range(1.0, 20.0, 0.2),  # 1.0到20.0，步长0.2 → 96个值
    "first_stage_token_limit": list(range(20, 101, 10)) if not FIXED_TOKEN_LIMIT else [0],  # 20-100步长10 或 固定0
    "k": list(range(8, 13, 1)),  # 8-12步长1 → 5个值
    "voting_metric": ["confidence", "self_certainty"],  # 2种指标
    "keyword_weight_multiplier": generate_range(10.0, 20.0, 0.5),  # 10-20步长0.5 → 21个值
    "keyword_window_size": list(range(10, 31, 1))  # 10-30步长1 → 21个值
}

# 固定参数
FIXED_PARAMS = {
    "max_items": 1000 if FIXED_MAX_ITEMS else None,  # 固定1000或参与循环
    "save_interval": 10,
}

# 并行配置
MAX_PARALLEL_JOBS = 8  # 最大并行任务数（10核Mac可以使用8并行）

# 计算总实验数
def _count_experiments():
    total = 1
    for key, values in PARAMETER_GRID.items():
        total *= len(values)
    return total

TOTAL_EXPERIMENTS = _count_experiments()

print(f"注意: 当前参数空间包含 {TOTAL_EXPERIMENTS:,} 个配置组合!")
print(f"预计耗时 (假设每个3分钟, {MAX_PARALLEL_JOBS}并行): {TOTAL_EXPERIMENTS / MAX_PARALLEL_JOBS * 3 / 60:.1f} 小时")

# 输出文件
RESULTS_FILE = "parameter_search_results.json"
LOCK_FILE = "parameter_search.lock"


# ==================== 核心函数 ====================

def generate_all_combinations() -> List[Dict[str, Any]]:
    """生成所有参数组合"""
    keys = list(PARAMETER_GRID.keys())
    values = [PARAMETER_GRID[k] for k in keys]
    
    combinations = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        # 将FIXED_PARAMS中的非None值添加到配置
        for key, value in FIXED_PARAMS.items():
            if value is not None:
                config[key] = value
        combinations.append(config)
    
    return combinations


def run_single_experiment(config: Dict[str, Any], input_file: str, 
                         exp_id: int, total_exp: int) -> Dict[str, Any]:
    """
    运行单个实验
    
    Args:
        config: 参数配置
        input_file: 输入数据文件
        exp_id: 实验ID
        total_exp: 总实验数
        
    Returns:
        实验结果字典
    """
    start_time = time.time()
    
    # 构建输出文件名（带时间戳和参数标识）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_abbr = "conf" if config['voting_metric'] == "confidence" else "sc"
    param_str = f"{metric_abbr}_vp{config['voting_power']}_n{config['first_stage_token_limit']}_m{config['k']}_k{config['keyword_weight_multiplier']}_s{config['keyword_window_size']}"
    output_file = f"results/exp_{exp_id}_{param_str}_{timestamp}.json"
    
    # 确保输出目录存在
    os.makedirs("results", exist_ok=True)
    
    # 构建命令
    cmd = [
        sys.executable,  # python
        "numerical_calculator.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--run_prunning_voting",
        "--voting_power", str(config["voting_power"]),
        "--first_stage_token_limit", str(config["first_stage_token_limit"]),
        "--k", str(config["k"]),
        "--keyword_weight_multiplier", str(config["keyword_weight_multiplier"]),
        "--keyword_window_size", str(config["keyword_window_size"]),
        "--voting_metric", config["voting_metric"],
        "--save_interval", str(config["save_interval"]),
    ]
    
    if config["max_items"] is not None:
        cmd.extend(["--max_items", str(config["max_items"])])
    
    print(f"\n[{exp_id}/{total_exp}] 启动实验: {param_str}")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        # 运行子进程
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        if result.returncode != 0:
            print(f"[{exp_id}/{total_exp}] 实验失败: {param_str}")
            print(f"错误输出: {result.stderr}")
            return {
                "exp_id": exp_id,
                "config": config,
                "status": "failed",
                "error": result.stderr,
                "duration": time.time() - start_time
            }
        
        # 解析结果
        accuracy, total_tokens, total_questions = parse_results(output_file)
        
        duration = time.time() - start_time
        
        result_dict = {
            "exp_id": exp_id,
            "config": config,
            "status": "success",
            "accuracy": accuracy,
            "total_tokens": total_tokens,
            "total_questions": total_questions,
            "duration": duration,
            "output_file": output_file,
            "timestamp": timestamp
        }
        
        print(f"[{exp_id}/{total_exp}] 完成: {param_str} | 准确率: {accuracy:.2%} | Token: {total_tokens} | 耗时: {duration:.1f}s")
        
        return result_dict
        
    except subprocess.TimeoutExpired:
        print(f"[{exp_id}/{total_exp}] 超时: {param_str}")
        return {
            "exp_id": exp_id,
            "config": config,
            "status": "timeout",
            "duration": time.time() - start_time
        }
    except Exception as e:
        print(f"[{exp_id}/{total_exp}] 异常: {param_str} - {str(e)}")
        return {
            "exp_id": exp_id,
            "config": config,
            "status": "error",
            "error": str(e),
            "duration": time.time() - start_time
        }


def parse_results(output_file: str) -> Tuple[float, int, int]:
    """
    解析实验结果文件，提取准确率和token总数
    
    Args:
        output_file: 结果文件路径
        
    Returns:
        (准确率, token总数, 总问题数)
    """
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            return 0.0, 0, 0
        
        # 统计准确率
        correct_count = sum(1 for item in data if item.get("is_correct", False))
        total_questions = len(data)
        accuracy = correct_count / total_questions if total_questions > 0 else 0.0
        
        # 统计token总数
        total_tokens = sum(item.get("token_count", 0) for item in data)
        
        return accuracy, total_tokens, total_questions
        
    except Exception as e:
        print(f"解析结果文件失败: {output_file} - {str(e)}")
        return 0.0, 0, 0


def save_result_atomic(result: Dict[str, Any], results_file: str):
    """
    原子性地追加单个结果到文件（使用文件锁）
    
    Args:
        result: 实验结果
        results_file: 结果文件路径
    """
    lock_path = LOCK_FILE
    
    # 简单的文件锁机制
    max_wait = 30
    waited = 0
    while os.path.exists(lock_path) and waited < max_wait:
        time.sleep(0.1)
        waited += 0.1
    
    try:
        # 创建锁文件
        with open(lock_path, 'w') as f:
            f.write(str(os.getpid()))
        
        # 读取现有结果
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = []
        else:
            results = []
        
        # 追加新结果
        results.append(result)
        
        # 写回文件
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
    finally:
        # 释放锁
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except:
                pass


def print_progress_bar(current: int, total: int, width: int = 50):
    """打印进度条"""
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    print(f'\r进度: [{bar}] {current}/{total} ({percent:.1%})', end='', flush=True)


def run_parameter_search(input_file: str, max_parallel: int = MAX_PARALLEL_JOBS):
    """
    运行参数搜索主流程
    
    Args:
        input_file: 输入数据文件
        max_parallel: 最大并行任务数
    """
    print("="*80)
    print("参数搜索启动")
    print("="*80)
    
    # 生成所有组合
    combinations = generate_all_combinations()
    total_exp = len(combinations)
    
    print(f"\n参数空间:")
    for key, values in PARAMETER_GRID.items():
        print(f"  {key}: {values}")
    print(f"\n总实验数: {total_exp}")
    print(f"并行任务数: {max_parallel}")
    print(f"结果文件: {RESULTS_FILE}")
    print("\n" + "="*80)
    
    # 确认继续
    response = input("\n是否继续? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("已取消")
        return
    
    # 初始化结果文件
    if os.path.exists(RESULTS_FILE):
        backup_file = f"{RESULTS_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(RESULTS_FILE, backup_file)
        print(f"\n已备份旧结果文件到: {backup_file}")
    
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)
    
    start_time = time.time()
    completed = 0
    
    # 使用进程池并行执行
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # 提交所有任务
        future_to_config = {
            executor.submit(run_single_experiment, config, input_file, idx+1, total_exp): (idx+1, config)
            for idx, config in enumerate(combinations)
        }
        
        # 收集结果
        for future in as_completed(future_to_config):
            exp_id, config = future_to_config[future]
            
            try:
                result = future.result()
                
                # 原子性保存结果
                save_result_atomic(result, RESULTS_FILE)
                
                completed += 1
                print_progress_bar(completed, total_exp)
                
            except Exception as e:
                print(f"\n实验 {exp_id} 执行异常: {str(e)}")
                completed += 1
    
    print("\n\n" + "="*80)
    print("参数搜索完成!")
    print("="*80)
    
    total_duration = time.time() - start_time
    print(f"总耗时: {total_duration/60:.1f} 分钟")
    print(f"结果已保存到: {RESULTS_FILE}")
    
    # 生成总结报告
    generate_summary_report(RESULTS_FILE)


def generate_summary_report(results_file: str):
    """生成总结报告"""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            print("无结果数据")
            return
        
        # 统计
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") != "success"]
        
        print(f"\n总结:")
        print(f"  成功: {len(successful)}/{len(results)}")
        print(f"  失败: {len(failed)}/{len(results)}")
        
        if successful:
            # 找到最佳配置
            best_result = max(successful, key=lambda x: x.get("accuracy", 0))
            
            print(f"\n最佳配置:")
            print(f"  准确率: {best_result['accuracy']:.2%}")
            print(f"  Token总数: {best_result['total_tokens']}")
            print(f"  参数:")
            for key, value in best_result['config'].items():
                if key in PARAMETER_GRID:
                    print(f"    {key}: {value}")
            
            # 保存最佳配置
            best_config_file = "best_config.json"
            with open(best_config_file, 'w', encoding='utf-8') as f:
                json.dump(best_result, f, indent=2, ensure_ascii=False)
            print(f"\n最佳配置已保存到: {best_config_file}")
    
    except Exception as e:
        print(f"生成报告失败: {str(e)}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="参数搜索脚本")
    parser.add_argument("--input_file", type=str, required=True, help="输入数据文件")
    parser.add_argument("--max_parallel", type=int, default=MAX_PARALLEL_JOBS, help="最大并行任务数")
    parser.add_argument("--dry_run", action="store_true", help="仅打印配置，不实际运行")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    if args.dry_run:
        # 仅显示将要运行的配置
        combinations = generate_all_combinations()
        print(f"总实验数: {len(combinations)}")
        print("\n前5个配置示例:")
        for i, config in enumerate(combinations[:5]):
            print(f"\n配置 {i+1}:")
            for key, value in config.items():
                if key in PARAMETER_GRID:
                    print(f"  {key}: {value}")
        print("\n...")
        return
    
    run_parameter_search(args.input_file, args.max_parallel)


if __name__ == "__main__":
    main()

