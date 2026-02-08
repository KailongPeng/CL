import os
import json
import glob
import pandas as pd
import numpy as np

# ====== 配置区域 ======
# 指向你的 predictions 文件夹 (infer_seq_qwen.sh 中设置的 PRED_OUTPUT_DIR)
PRED_DIR = "/path/to/your/outputs_LLM-CL/debug_test/qwen/predictions" 
# =====================

def load_metrics(pred_dir):
    # 查找所有的结果文件
    files = glob.glob(os.path.join(pred_dir, "results-*-*-*.json"))
    
    data = []
    tasks_set = set()
    
    print(f"找到 {len(files)} 个结果文件。")
    
    for f_path in files:
        filename = os.path.basename(f_path)
        # 解析文件名: results-{round}-{task_id}-{task_name}.json
        try:
            parts = filename.replace("results-", "").replace(".json", "").split("-")
            train_round = int(parts[0])
            test_task_id = int(parts[1])
            task_name = "-".join(parts[2:])
            
            with open(f_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
            # 尝试提取指标，不同数据集指标名称可能不同 (acc, accuracy, f1 等)
            eval_res = content.get("eval", {})
            score = 0
            metric_name = "unknown"
            
            # 优先级提取常见的指标
            for key in ["accuracy", "acc", "f1", "rouge-L", "exact_match"]:
                if key in eval_res:
                    score = eval_res[key]
                    metric_name = key
                    break
            
            # 如果是百分比字符串，转换为浮点数
            if isinstance(score, str) and "%" in score:
                score = float(score.strip("%")) / 100.0
                
            data.append({
                "Train_Round": train_round,
                "Test_Task": task_name,
                "Test_Task_ID": test_task_id,
                "Score": score,
                "Metric": metric_name
            })
            tasks_set.add((test_task_id, task_name))
            
        except Exception as e:
            print(f"解析文件 {filename} 失败: {e}")

    return data, sorted(list(tasks_set))

def calculate_cl_metrics(data, task_list):
    if not data:
        print("没有数据可分析。")
        return

    df = pd.DataFrame(data)
    
    # 创建矩阵 R[i, j]: 在 Round i 训练后，在 Task j 上的得分
    rounds = df['Train_Round'].unique()
    rounds.sort()
    
    max_round = rounds[-1]
    num_tasks = len(task_list)
    
    # 初始化矩阵 (行=训练轮次, 列=测试任务)
    matrix = pd.DataFrame(index=rounds, columns=[t[1] for t in task_list])
    
    for _, row in df.iterrows():
        matrix.loc[row['Train_Round'], row['Test_Task']] = row['Score']

    print("\n>>> 持续学习性能矩阵 (Accuracy Matrix):")
    print(matrix.round(4))
    print("-" * 50)

    # 计算指标
    # 1. Average Accuracy (最后一轮的平均分)
    last_round_scores = matrix.loc[max_round].dropna()
    avg_acc = last_round_scores.mean()
    
    # 2. Forgetting Measure (遗忘率)
    # F_j = max(之前所有轮次该任务的得分) - 当前轮次该任务得分
    forgetting = []
    for task_idx, (tid, task_name) in enumerate(task_list):
        if task_idx > max_round: continue
        
        # 获取该任务在历史上的最高分 (通常是刚训练完那一轮，但取 max 更稳健)
        # 只看直到 max_round 之前的历史
        history_scores = matrix[task_name].loc[:max_round]
        peak_score = history_scores.max()
        current_score = matrix.loc[max_round, task_name]
        
        if pd.notna(peak_score) and pd.notna(current_score):
            forgetting.append(peak_score - current_score)

    avg_forgetting = np.mean(forgetting) if forgetting else 0.0

    print(f"\n>>> 分析报告 (Round {max_round}):")
    print(f"1. 平均精度 (Avg Accuracy): {avg_acc:.4f}")
    print(f"2. 平均遗忘 (Avg Forgetting): {avg_forgetting:.4f} (越低越好)")
    print("   * 正数表示遗忘，负数表示性能提升(Backward Transfer)")

if __name__ == "__main__":
    # 确保这里的路径和你脚本里的一致
    # 你可以通过命令行参数传入，或者直接在这里修改
    data, tasks = load_metrics(PRED_DIR)
    calculate_cl_metrics(data, tasks)