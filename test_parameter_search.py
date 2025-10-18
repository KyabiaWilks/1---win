#!/usr/bin/env python3
"""
参数搜索测试脚本 - 快速验证流程
使用极小的参数空间测试整个流程是否正常
"""

import sys
import os

# 临时修改参数空间为仅在 voting_power 上循环，其余参数固定
test_config = """
PARAMETER_GRID = {
    "voting_power": list(range(1, 21))  # 1 到 20，步长为 1
}

FIXED_PARAMS = {
    "max_items": 492,  # 测试492个问题
    "save_interval": 1,
    "voting_metric": "confidence",  # 固定使用信心度
    # 固定其余参数
    "first_stage_token_limit": 50,
    "k": 16,
    "keyword_weight_multiplier": 10.0,
    "keyword_window_size": 40,
}

MAX_PARALLEL_JOBS = 8  # 8并行
"""

def create_test_config():
    """创建测试配置，自动定位源脚本位置并写入同目录测试脚本。

    返回: 创建的测试脚本路径(str)；失败则返回 None
    """
    # 寻找源脚本位置（兼容根目录与 parameter_search_tools/）
    candidate_paths = [
        os.path.join("", "parameter_search.py"),
        os.path.join("parameter_search_tools", "parameter_search.py"),
    ]

    source_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            source_path = p
            break

    if not source_path:
        return None

    with open(source_path, 'r') as f:
        original_content = f.read()

    # 备份原始配置
    with open(source_path + ".backup", 'w') as f:
        f.write(original_content)

    # 替换参数空间定义块
    lines = original_content.split('\n')
    new_lines = []
    skip = False
    for line in lines:
        if 'PARAMETER_GRID = {' in line:
            skip = True
            new_lines.append(test_config)
        elif skip and 'MAX_PARALLEL_JOBS' in line:
            skip = False
        elif not skip:
            new_lines.append(line)

    # 测试脚本与源文件放在同一目录
    target_dir = os.path.dirname(source_path)
    test_script_path = os.path.join(target_dir if target_dir else ".", "parameter_search_test.py")

    with open(test_script_path, 'w') as f:
        f.write('\n'.join(new_lines))

    rel_path = os.path.relpath(test_script_path)
    print(f"测试配置已创建: {rel_path}")
    print("测试参数空间: 20 个配置 (voting_power=1..20)")
    print("每个配置测试 492 个问题")
    print("并行数: 8")
    return test_script_path


def run_test(test_script_path: str):
    """运行测试"""
    import subprocess
    import json

    input_file = "test_data/mmlu-redux/openPangu-Embedded-7B_n64.json"

    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return False

    cmd = [
        sys.executable,
        test_script_path,
        "--input_file", input_file,
        "--max_parallel", "8",
    ]

    print(f"\n运行命令: {' '.join(cmd)}")
    print("="*80)

    try:
        result = subprocess.run(cmd, check=True)
        ok = result.returncode == 0
        if not ok:
            return False

        # 生成 power 汇总
        results_file = "parameter_search_results.json"
        summary = []
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except json.JSONDecodeError:
                results = []

            for r in results:
                cfg = r.get("config", {})
                # 仅统计我们这次运行配置（confidence + 固定其他参数）
                if cfg.get("voting_metric") != "confidence":
                    continue
                if cfg.get("first_stage_token_limit") != 50:
                    continue
                if cfg.get("k") != 16:
                    continue
                if float(cfg.get("keyword_weight_multiplier", 0)) != 10.0:
                    continue
                if cfg.get("keyword_window_size") != 40:
                    continue

                summary.append({
                    "voting_power": cfg.get("voting_power"),
                    "accuracy": r.get("accuracy"),
                    "total_tokens": r.get("total_tokens"),
                    "total_questions": r.get("total_questions"),
                    "output_file": r.get("output_file"),
                    "status": r.get("status")
                })

            os.makedirs("results", exist_ok=True)
            with open("results/power_summary.json", 'w', encoding='utf-8') as f:
                json.dump(sorted(summary, key=lambda x: x["voting_power"]), f, indent=2, ensure_ascii=False)

        return True
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {e}")
        return False


def main():
    print("="*80)
    print("参数搜索测试脚本")
    print("="*80)
    print("\n这将创建一个测试配置:")
    print("  - 20个参数组合 (voting_power=1..20, 步长1)")
    print("  - 每个测试492个问题")
    print("  - 8个并行任务")
    print("  - 预计耗时取决于模型速度和硬件")
    print("\n这不会影响原始的 parameter_search.py")
    
    response = input("\n是否继续? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("已取消")
        return
    
    # 创建测试配置
    test_script_path = create_test_config()
    if not test_script_path:
        print("创建测试配置失败")
        return

    # 运行测试
    success = run_test(test_script_path)
    
    if success:
        print("\n" + "="*80)
        print("测试完成!")
        print("="*80)
        print("\n结果文件:")
        print("  - parameter_search_results.json (主结果文件)")
        print("  - best_config.json (最佳配置)")
        print("  - results/exp_*.json (各实验详细输出)")
        print("\n分析结果:")
        print("  python analyze_search_results.py")
    else:
        print("\n测试失败")
    
    # 清理
    if test_script_path and os.path.exists(test_script_path):
        os.remove(test_script_path)
        print("\n已清理测试文件")


if __name__ == "__main__":
    main()

