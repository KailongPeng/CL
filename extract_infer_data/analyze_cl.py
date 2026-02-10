# D:\Desktop\files\huawei\repo\continual_learning\TRACE\extract_infer_data\analyze_cl.py
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
                
            # 尝试提取指标
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
    
    # 获取所有的训练轮次
    rounds = sorted(df['Train_Round'].unique())
    current_round = rounds[-1] # 当前所在的最后一轮 (t)
    
    # 建立任务ID到名称的映射，方便后续查找
    id_to_task = {t[0]: t[1] for t in task_list}
    
    # 初始化矩阵 (行=训练轮次, 列=测试任务名称)
    # 确保列的顺序按照 Task ID 排序
    sorted_task_names = [t[1] for t in task_list]
    matrix = pd.DataFrame(index=rounds, columns=sorted_task_names)
    
    for _, row in df.iterrows():
        matrix.loc[row['Train_Round'], row['Test_Task']] = row['Score']

    # 按照 Task ID 排序矩阵的列，保证视觉上的对角线逻辑
    matrix = matrix[sorted_task_names]

    print(f"\n>>> 持续学习性能矩阵 (Round {current_round}):")
    # 打印时填充NaN以便查看，保留4位小数
    print(matrix.round(4).fillna("-"))
    print("-" * 65)

    # ==========================================
    # 1. 计算 Average (OP_t)
    # 公式: 当前轮次(t)下，所有已学任务(1到t)的平均分
    # ==========================================
    
    # 获取最后一行的数据
    current_scores = matrix.loc[current_round]
    
    # 我们只关心直到当前轮次涉及的任务。
    # 假设: Task ID 与 Round 一一对应 (即 Round 1 训练 Task 1)
    # 如果你的实验设置里 Round 1 训练了 Task A, B... 需要根据实际情况调整切片
    # 这里假设是标准的 TRACE 设置：Round t 对应 Task ID t
    tasks_up_to_now = [id_to_task[r] for r in rounds if r in id_to_task]
    
    # 过滤掉矩阵中可能存在的 NaN (未测试的任务)
    valid_scores = current_scores[tasks_up_to_now].dropna()
    
    op_score = valid_scores.mean()

    # ==========================================
    # 2. 计算 BWT (Backward Transfer)
    # 公式: Sum(R_{t,i} - R_{i,i}) / (t-1)
    # 其中 i < t (旧任务)
    # ==========================================
    
    bwt_sum = 0
    bwt_count = 0
    bwt_details = []

    # 遍历之前的每一轮 i (从第1轮 到 t-1轮)
    # 注意：rounds 是列表，rounds[:-1] 排除了当前轮
    prev_rounds = [r for r in rounds if r < current_round]
    
    if not prev_rounds:
        print("当前为第 1 轮，无法计算 BWT (没有旧任务)。")
        bwt_score = 0.0
    else:
        print(f"{'Task':<20} | {'Original (R_ii)':<15} | {'Current (R_ti)':<15} | {'Diff'}")
        print("-" * 65)
        
        for r_i in prev_rounds:
            # 找到 Round i 对应的任务名称 (Task i)
            # 假设 Round i 训练的是 Task ID i
            if r_i not in id_to_task:
                continue
                
            task_name_i = id_to_task[r_i]
            
            # 获取 R_{i,i}: 当初刚学完该任务时的分数 (对角线)
            r_ii = matrix.loc[r_i, task_name_i]
            
            # 获取 R_{t,i}: 现在(第t轮)该任务的分数 (最后一列)
            r_ti = matrix.loc[current_round, task_name_i]
            
            if pd.notna(r_ii) and pd.notna(r_ti):
                diff = r_ti - r_ii
                bwt_sum += diff
                bwt_count += 1
                
                print(f"{task_name_i:<20} | {r_ii:.4f}          | {r_ti:.4f}          | {diff:.4f}")
            else:
                print(f"{task_name_i:<20} | 数据缺失 (NaN) - 跳过")

        # 计算平均值
        bwt_score = bwt_sum / bwt_count if bwt_count > 0 else 0.0

    print("-" * 65)
    print(f">>> 最终指标汇总 (Round {current_round}):")
    print(f"1. Average (OP) : {op_score:.4f}  (当前所有已学任务的平均分)")
    print(f"2. BWT          : {bwt_score:.4f}  (旧任务平均遗忘程度，负数代表遗忘)")

if __name__ == "__main__":
    # 加载数据
    data, tasks = load_metrics(PRED_DIR)
    # 计算 BWT 和 Average
    calculate_cl_metrics(data, tasks)