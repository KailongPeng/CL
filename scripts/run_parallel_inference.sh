#!/bin/bash

# ================= 配置区域 =================
# 1. 定义你要跑的所有数据集
TASK_LIST=("C-STANCE" "FOMC" "MeetingBank" "Py150" "ScienceQA" "NumGLUE-cm" "NumGLUE-ds" "20Minuten")

# 2. 定义允许使用的 GPU ID (根据你的要求)
GPU_IDS=(0 2 3 4)

# 3. 基础参数配置 (请根据你的实际路径修改)
BASE_MODEL="D:\Desktop\files\huawei\repo\continual_learning\TRACE\Qwen-0.6B"
# 注意：这里假设你的 inference_model_path 是训练输出目录
TRAIN_OUTPUT_DIR="D:\Desktop\files\huawei\repo\continual_learning\TRACE\outputs_LLM-CL\debug_test\qwen"
PRED_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}/predictions"
DATA_PATH="D:\Desktop\files\huawei\repo\continual_learning\TRACE\LLM-CL_Benchmark"

# 确保输出目录存在
mkdir -p "$PRED_OUTPUT_DIR"

# ================= 循环启动任务 =================
# 获取 GPU 的数量
NUM_GPUS=${#GPU_IDS[@]}

for i in "${!TASK_LIST[@]}"; do
    TASK=${TASK_LIST[$i]}
    
    # 计算当前任务应该分配给哪张 GPU (取模运算)
    GPU_INDEX=$((i % NUM_GPUS))
    CURRENT_GPU=${GPU_IDS[$GPU_INDEX]}
    
    echo "🚀 [启动任务] Dataset: $TASK ---> GPU ID: $CURRENT_GPU"

    # 核心命令：
    # 1. CUDA_VISIBLE_DEVICES=$CURRENT_GPU : 限制当前进程只能看到这一张卡
    # 2. & : 在后台运行，不阻塞脚本继续执行下一个循环
    
    CUDA_VISIBLE_DEVICES=$CURRENT_GPU python inference/infer_single.py \
        --data_path "$DATA_PATH" \
        --inference_tasks "$TASK" \
        --model_name_or_path "$BASE_MODEL" \
        --inference_model_path "$TRAIN_OUTPUT_DIR" \
        --inference_batch 1 \
        --max_prompt_len 1024 \
        --max_ans_len 512 \
        --seed 42 \
        --CL_method lora \
        --inference_output_path "$PRED_OUTPUT_DIR" \
        --local_rank 0 \
        > "${PRED_OUTPUT_DIR}/log_${TASK}.log" 2>&1 &
        
    # 保存 PID 以便后续管理（可选）
    pids[${i}]=$!
    
    # 稍微暂停 2 秒，防止瞬间并发导致文件读取 IO 冲突
    sleep 2
done

# ================= 等待结束 =================
echo "🎉 所有任务已在后台启动！正在并行处理中..."
echo "请查看 ${PRED_OUTPUT_DIR} 下的 log_*.log 文件监控进度。"
echo "正在等待所有子进程结束..."

# wait 命令会挂起当前脚本，直到所有后台任务（&）都执行完毕
wait

echo "✅ 所有推理任务执行完毕！"