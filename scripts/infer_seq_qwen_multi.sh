# D:\Desktop\files\huawei\repo\continual_learning\TRACE\scripts\infer_seq_qwen_multi.sh
#!/bin/bash
# 随机生成端口
port=$(shuf -i 25000-30000 -n 1)

# ====== 👇 请修改这里 (保持与 train 脚本一致) 👇 ======
# 1. 原始底座模型路径
tag="qwen"
# if [ "$tag" == "qwen" ]; then
BASE_MODEL_PATH="/path/to/your/Qwen-0.6B" 
# else
#     BASE_MODEL_PATH="/path/to/your/memorized_qwen" 
# fi
# 2. 数据集路径
DATA_PATH="/path/to/your/LLM-CL_Benchmark"
# 3. 刚才训练的输出目录 (脚本会自动去下面找 /0, /1 等文件夹)
TRAIN_OUTPUT_DIR="/path/to/your/outputs_LLM-CL/debug_test/${tag}/"
# ====================================================

# 推理结果保存的位置 
PRED_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}/predictions"
mkdir -p $PRED_OUTPUT_DIR

# 临时数据文件路径 (infer_multi 需要)
DATA_OUTPUT_PATH="${TRAIN_OUTPUT_DIR}/data_files"
mkdir -p $DATA_OUTPUT_PATH

echo ">>> 开始多卡分布式推理..."
echo ">>> 底座模型: $BASE_MODEL_PATH"
echo ">>> 加载微调权重目录: $TRAIN_OUTPUT_DIR"

# 关键修改点：
# 1. 脚本文件: inference/infer_single.py -> inference/infer_multi.py
# 2. 参数名: --inference_tasks -> --dataset_name (这是 infer_multi.py 定义的参数名)
# 3. 增加: --data_output_path (infer_multi 需要存放处理后的数据缓存)
# 4. --inference_batch: 多卡推理可以使用稍大一点的 batch (例如 4 或 8)，取决于显存
# 5. --CL_method: 注意查看下方的【重要提示】

deepspeed --include localhost:1,2 --master_port $port inference/infer_multi.py \
    --data_path $DATA_PATH \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds \
    --model_name_or_path $BASE_MODEL_PATH \
    --inference_model_path ${TRAIN_OUTPUT_DIR} \
    --inference_batch 4 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 42 \
    --deepspeed \
    --data_output_path $DATA_OUTPUT_PATH \
    --CL_method lora \
    --inference_output_path $PRED_OUTPUT_DIR > $TRAIN_OUTPUT_DIR/infer_multi.log 2>&1 &

echo ">>> 推理任务已提交！请立即执行下面这行命令查看日志："
echo "tail -f $TRAIN_OUTPUT_DIR/infer_multi.log"