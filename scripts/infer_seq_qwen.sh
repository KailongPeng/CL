# D:\Desktop\files\huawei\repo\continual_learning\TRACE\scripts\infer_seq.sh
# #!bin/bash
# port=$(shuf -i25000-30000 -n1)
# deepspeed --include=localhost:0 --master_port $port inference/infer_single.py  \
#     --data_path /mnt/data/user/zhang_yuansen/LLM-CL_Benchmark \
#     --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /mnt/data/user/zhang_yuansen/PTMs/llama-2-7b \
#     --inference_model_path /mnt/data/user/zhang_yuansen/outputs_LLM-CL/naive \
#     --inference_batch 4 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --seed 1234 \
#     --deepspeed \
#     --CL_method base \
#     --inference_output_path /mnt/data/user/zhang_yuansen/outputs_LLM-CL/naive/predictions > /mnt/data/user/zhang_yuansen/outputs_LLM-CL/naive/infer.log 2>&1 &


# D:\Desktop\files\huawei\repo\continual_learning\TRACE\scripts\infer_seq.sh
#!/bin/bash
# 随机生成端口
port=$(shuf -i25000-30000 -n1)

# ====== 👇 请修改这里 (保持与 train 脚本一致) 👇 ======
# 1. 原始底座模型路径
tag="qwen"
if [ "$tag" == "qwen" ]; then
    BASE_MODEL_PATH="/path/to/your/Qwen-0.6B" 
else
    BASE_MODEL_PATH="/path/to/your/memorized_qwen" 
fi
# 2. 数据集路径
DATA_PATH="/path/to/your/LLM-CL_Benchmark"
# 3. 刚才训练的输出目录 (脚本会自动去下面找 /0 文件夹)
TRAIN_OUTPUT_DIR="/path/to/your/outputs_LLM-CL/debug_test/${tag}/"
# ====================================================

# 推理结果保存的位置 
PRED_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}/predictions"
mkdir -p $PRED_OUTPUT_DIR

echo ">>> 开始推理冒烟测试..."
echo ">>> 底座模型: $BASE_MODEL_PATH"
echo ">>> 加载微调权重: $TRAIN_OUTPUT_DIR/0"

# 关键修改点说明：
# 1. --include localhost:7 : 指定使用 7 号卡
# 2. --inference_tasks C-STANCE : 只测 C-STANCE (对应刚才的训练)
# 3. --inference_model_path : 指向 ${TRAIN_OUTPUT_DIR}/0 (因为你之前的日志显示保存到了 .../0)
# 4. --inference_batch 1 : 设为 1 保证显存安全，验证通过后再调大

deepspeed --include localhost:4,5,6,7 --master_port $port inference/infer_single.py \
    --data_path $DATA_PATH \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path $BASE_MODEL_PATH \
    --inference_model_path ${TRAIN_OUTPUT_DIR} \
    --inference_batch 1 \
    --max_prompt_len 2048 \
    --max_ans_len 512 \
    --seed 42 \
    --deepspeed \
    --CL_method base \
    --inference_output_path $PRED_OUTPUT_DIR > $TRAIN_OUTPUT_DIR/infer.log 2>&1 &

echo ">>> 推理任务已提交！请立即执行下面这行命令查看日志："
echo "tail -f $TRAIN_OUTPUT_DIR/infer.log"