# D:\Desktop\files\huawei\repo\continual_learning\TRACE\scripts\train_seq_naive.sh

# port=$(shuf -i25000-30000 -n1)
# deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $port training/main.py  \
#     --data_path /mnt/data/user/zhang_yuansen/LLM-CL_Benchmark \
#     --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /mnt/data/user/zhang_yuansen/PTMs/llama-2-7b-chat \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 16 \
#     --max_prompt_len 2048 \
#     --max_ans_len 512 \
#     --learning_rate 1e-5 \
#     --weight_decay 0. \
#     --num_train_epochs 5,3,7,5,3,5,5,7 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --num_warmup_steps 0 \
#     --seed 1234 \
#     --zero_stage 2 \
#     --deepspeed \
#     --print_loss \
#     --CL_method base \
#     --output_dir /mnt/data/user/zhang_yuansen/outputs_LLM-CL/naive > /mnt/data/user/zhang_yuansen/outputs_LLM-CL/naive/train.log 2>&1 &


# D:\Desktop\files\huawei\repo\continual_learning\TRACE\scripts\train_seq_naive.sh
#!/bin/bash
# 随便生成一个端口
port=$(shuf -i25000-30000 -n1)

# ====== 👇 请修改这里 👇 ======
tag="qwen"
if [ "$tag" == "qwen" ]; then
    MODEL_PATH="/path/to/your/Qwen-0.6B" 
else
    MODEL_PATH="/path/to/your/memorized_qwen" 
fi
DATA_PATH="/path/to/your/LLM-CL_Benchmark"
OUTPUT_DIR="/path/to/your/outputs_LLM-CL/debug_test/${tag}"
# ==============================

mkdir -p $OUTPUT_DIR

echo ">>> 开始冒烟测试..."
echo ">>> 目标：验证模型加载、Tokenizer修复、DeepSpeed启动是否正常"

deepspeed --include localhost:7 --master_port $port training/main.py \
    --data_path $DATA_PATH \
    --dataset_name C-STANCE \
    --model_name_or_path $MODEL_PATH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_prompt_len 256 \
    --max_ans_len 128\
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 42 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method lora \
    --num_sinks 0 \
    --use_sink False \
    --sliding_window 2048 \
    --segment_size 2048 \
    --output_dir $OUTPUT_DIR > $OUTPUT_DIR/train.log 2>&1 &


echo ">>> 任务已提交！请立即执行下面这行命令查看日志："
echo "tail -f $OUTPUT_DIR/train.log"


# 150 服务器， 环境是mllm_kailong
