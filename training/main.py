# D:\Desktop\files\huawei\repo\continual_learning\TRACE\training\main.py
#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import sys
sys.dont_write_bytecode = True

import argparse
import os
import math
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    get_constant_schedule_with_warmup
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import safe_get_full_grad


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model

# ================== 自定义模型注册 ==================
from transformers import AutoConfig, AutoModelForCausalLM
from models.memories.modeling_memory import LaCTQwen3Model
from models.configuration_qwen import MemorizedQwenConfig

AutoConfig.register("memorized_qwen", MemorizedQwenConfig)
AutoModelForCausalLM.register(MemorizedQwenConfig, LaCTQwen3Model)
print("✅ 已注册自定义模型：MemorizedQwenConfig -> LaCTQwen3Model")

# add flash attention
from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn

# replace_llama_attn_with_flash_attn()
# replace_bloom_attn_with_flash_attn()

# my_peft中修改了lora相关的逻辑
from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import PP, convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model


from params import Method2Class, AllDatasetName


# TODO, check support for OPT and llama


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset, a single data path.')
    parser.add_argument('--dataset_name',
                        type=list_of_strings,
                        default='all',
                        help='Dataset to be used.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=list_of_strings,
                        default=None,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
    # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    # store_true 表示如果命令行中有这个参数，则 args.disable_dropout 为 True, 否则默认为 False
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    # added by wangxiao
    parser.add_argument('--CL_method',
                default=None,
                help='continual learning method used')
    parser.add_argument("--num_sinks", type=int, default=0, help="Number of sink tokens.")
    parser.add_argument("--use_sink", type=str, default="False", help="Whether to use attention sink (True/False).")
    parser.add_argument("--sliding_window", type=int, default=2048, help="Size of the sliding window.")
    parser.add_argument("--segment_size", type=int, default=2048, help="Size of the memory segment.")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()


    return args


def main():
    args = parse_args()
    
    # ================= 🚨 新增：开启异常检测 🚨 =================
    # 这会降低运行速度，但能帮你找到导致 NaN 的确切代码行（例如前向传播里的数学错误）
    torch.autograd.set_detect_anomaly(True) 
    # ==========================================================

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="v2_sft")
    
    # ================= 🚨 [新增修改] 强制使用 FP32 🚨 =================
    # 无论脚本参数怎么传，这里强制关闭 fp16 和 bf16
    # 这是解决 "!!!!!!" 输出和 Loss NaN 的终极手段
    print("\n" + "!"*40)
    # print("⚠️  正在强制修改 DeepSpeed 配置为 FP32 (Full Precision)...")
    
    if "fp16" not in ds_config: ds_config["fp16"] = {}
    ds_config["fp16"]["enabled"] = False

    if "bf16" not in ds_config: ds_config["bf16"] = {}
    ds_config["bf16"]["enabled"] = True
    ds_config["bfloat16"] = {"enabled": True}
    
    # print(f"✅ FP16/BF16 已禁用。当前精度模式: FP32 (Float32)")
    print("!"*40 + "\n")
    # ==================================================================

    # set batch size
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # Barrier to make sure all process are ready to train
    torch.distributed.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)

    # 将 Padding ID 设为 151643 (<|endoftext|>)，彻底避开 <|im_end|>
    tokenizer.pad_token_id = 151643
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)

    # 确保 EOS ID 是正确的
    tokenizer.eos_token_id = 151645 # <|im_end|>

    # # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"
    # 强制改为右填充 (Right Padding) 用于训练
    # tokenizer.padding_side = 'right'  # ✅ 必须强制修改
    # tokenizer.truncation_side = 'right' # 通常配合 padding side 一起改
    # print(f"🔄 Padding Side 强制修正为: {tokenizer.padding_side}")

    # Qwen 补丁：如果没有 pad_token，将其设为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    


    # 这部分代码可以不要
    print("="*60)
    print(f"Loading MemorizedQwenConfig from {args.model_name_or_path}...")
    config = MemorizedQwenConfig.from_pretrained(args.model_name_or_path)
    print(f"config.model_type = {config.model_type}")
    print("="*60)
    # 这部分代码可以不要

    print(f"ds_config={ds_config}")

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=ds_config,
                            disable_dropout=args.disable_dropout,
                            args=args
                            )
    
    # # ================= 🚨 强制模型权重转 FP32 🚨 =================
    # # 你的日志显示 MA 1.41 GB，这是半精度的特征。
    # # 我们必须手动把模型转成 float()，让显存占用变成 2.5 GB 左右，才算成功。
    # print(f"🔄 [Before] 模型数据类型: {model.dtype}")
    
    # # 只要 DeepSpeed 配置禁用了 fp16/bf16，我们就强制转 float32
    # if not ds_config["fp16"]["enabled"] and not ds_config["bf16"]["enabled"]:
    #     print("⚠️ 正在执行强制 FP32 转换 (model.float())...")
    #     model = model.float()
        
    # print(f"✅ [After] 模型数据类型: {model.dtype}")
    # # ============================================================

    # some CL methods can be realized by peft
    if args.CL_method == "LFPT5":
        from utils.my_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, LoraConfig, TaskType

        initial_prompt = getInitialPrompt(tokenizer, prompt_token_number=300)
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=300,
            prompt_tuning_init_text=initial_prompt,
            tokenizer_name_or_path=args.model_name_or_path,
        )
        model = get_peft_model(model, peft_config)

    if args.CL_method == "O-LoRA":
        from utils.my_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, LoraConfig, TaskType

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = True
            elif name.find("lora_") != -1:
                param.requires_grad = False
                
    if args.CL_method == "OGD":
        from peft import get_peft_model, LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if name.find("lora") != -1:
                param.requires_grad = True

    if args.CL_method == "lora":
        from peft import get_peft_model, LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, # 告诉 PEFT “我现在要训练”，它会启用 Dropout，并确保梯度可以计算。
            r=8,  # 秩（Rank）。这是 LoRA 中最重要的参数，决定了外挂模块的“大小”和“容量”
            lora_alpha=16, # 典型的 2倍 r 设置，稳定. 缩放系数alpha  LoRA 更新权重的公式是 $$W_{new} = W_{old} + \frac{\alpha}{r} \cdot (A \times B)$$
            lora_dropout=0.05, # 在训练过程中，随机把 5% 的 LoRA 神经元输出置为 0。防止过拟合
            # target_modules=["gate_proj", "up_proj", "down_proj"], 
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_modules=["gate_proj", "up_proj", "down_proj","q_proj", "k_proj", "v_proj", "o_proj"], 
        )
        model = get_peft_model(model, peft_config)

        # 验证逻辑：确保只训练 LoRA 参数
        # ====== 增加显眼的打印格式 ======
        print("\n" + "="*50)
        print(f"✅ LoRA 配置生效！正在针对 {peft_config.target_modules} 进行训练")
        model.print_trainable_parameters()  # 预期结果：trainable params 应该在 1% - 5% 之间
        print("="*50 + "\n")
        # ==============================

        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    train_task_list = {}
    eval_task_list = {}
    test_task_list = {}


    if args.dataset_name[0] == "all":
        Datasets = AllDatasetName
    else:
        Datasets = args.dataset_name
    for dataset in Datasets:
        dataset_path = os.path.join(args.data_path,dataset)
        # Prepare the data
        train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
            args.local_rank,
            dataset_path,
            args.data_output_path,
            args.seed,
            tokenizer=tokenizer,
        )

        # DataLoaders creation:
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = SequentialSampler(eval_dataset)
            test_sampler = SequentialSampler(test_dataset)

        else:
            train_sampler = DistributedSampler(train_dataset)
            eval_sampler = DistributedSampler(eval_dataset)
            test_sampler = DistributedSampler(test_dataset)

        data_collator = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=False
        )
        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=True
        )
                

        train_dataloader = DataLoader(train_dataset,
                                    collate_fn=data_collator,
                                    sampler=train_sampler,
                                    batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset,
                                    collate_fn=data_collator,
                                    sampler=eval_sampler,
                                    batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset,
                            collate_fn=inf_data_collator,
                            sampler=test_sampler,
                            batch_size=args.per_device_eval_batch_size)
        train_task_list[dataset] = train_dataloader
        eval_task_list[dataset] = eval_dataloader
        test_task_list[dataset] = test_dataloader


    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                # TODO, check output
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    def get_optimizer(model):
        # Split weights in two groups, one with weight decay and the other not.
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.weight_decay)

        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
        optimizer = AdamOptimizer(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                betas=(0.9, 0.95))
        
        total_train_dataloader_len = sum(len(train_task_list[task]) for task in list(train_task_list.keys()))
        num_update_steps_per_epoch = math.ceil(
            total_train_dataloader_len / args.gradient_accumulation_steps)
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps
        )
        
        return optimizer, lr_scheduler
    
    if args.CL_method=="PP" or args.CL_method=="L2P":
        model_path_lower = args.model_name_or_path.lower()
        if "opt" in model_path_lower:
            embed_tokens = model.model.decoder.embed_tokens
        elif "llama" in model_path_lower or "qwen" in model_path_lower:
            embed_tokens = model.model.embed_tokens

        embed_tokens_shape = embed_tokens.weight.shape
        args.embed_tokens_dim = embed_tokens_shape[1]
        args.embed_tokens_length = embed_tokens_shape[0]
        args.embed_tokens = embed_tokens
            
        if args.CL_method=="PP":
            args.prefix_len = 20
            args.task_length = len(train_task_list)
            model = convert_PP_model(model, args)
            
        elif args.CL_method=="L2P":
            args.pool_size = 10
            args.prompt_length = 5
            args.prompt_init = "uniform"
            model = convert_L2P_model(model, args)
            for name, params in model.named_parameters():
                if "prompt" not in name:
                    params.requires_grad=False
                    
    optimizer, lr_scheduler = get_optimizer(model)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    # Initialize the global progress bar
    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    # ================= 新增：梯度监控钩子 =================
    # 这个函数会在每次反向传播计算梯度时被调用
    def log_grad_hook(name):
        def hook(grad):
            # 检查 NaN (Not a Number)
            if torch.isnan(grad).any():
                print(f"\n💀 [NaN DETECTED] Layer: {name}")
                print(f"   Shape: {grad.shape}")
                print(f"   Min: {grad.min()}, Max: {grad.max()}")
                # 可以在这里抛出异常强制停止，或者由 DeepSpeed 处理
            
            # 检查 Inf (无穷大，通常是梯度爆炸的前兆)
            elif torch.isinf(grad).any():
                print(f"\n💥 [Inf DETECTED] Layer: {name}")
                print(f"   Shape: {grad.shape}")
                print(f"   Min: {grad.min()}, Max: {grad.max()}")
            
            # 如果你想看正常的梯度统计（可选，会刷屏，建议仅在调试极个别 step 时开启）
            # else:
            #     if args.global_rank == 0:  # 只在主进程打印
            #         print(f"✅ {name} grad_mean: {grad.mean().item():.6f} | std: {grad.std().item():.6f}")
        return hook

    print("🔎 正在注册梯度监控钩子...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 只监控需要训练的层 (即 LoRA 层)
            print(f"   Watching gradient for: {name}")
            param.register_hook(log_grad_hook(name))
    print("🔎 钩子注册完成。\n")
    # ==========================================================

    if args.CL_method in Method2Class.keys():
        CL_Trainer = Method2Class[args.CL_method](model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        CL_Trainer.train_continual()


if __name__ == "__main__":
    import os
    import sys
    
    # ================= 🔧 调试模式专用配置 🔧 =================
    # 1. 只有在 VSCode Debug 或者是直接运行 Python 时才生效
    #    如果你在服务器用 sh 脚本跑，不受影响（因为会有参数覆盖）
    if len(sys.argv) == 1:  # 没有命令行参数，说明是手动点的运行
        print("🚀 进入 VSCode 单卡 Debug 模式 (模拟 DeepSpeed 环境)...")

        # --- A. 伪造 DeepSpeed 分布式环境变量 (欺骗 DeepSpeed 以为在分布式运行) ---
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"          # 我是主进程
        os.environ["LOCAL_RANK"] = "0"    # 我是当前节点的第0张卡
        os.environ["WORLD_SIZE"] = "1"    # 全世界只有我这1个进程
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 只用第1张显卡

        # --- B. 伪造命令行参数 (把 shell 脚本里的参数搬过来) ---
        # 请根据你的实际路径修改下面三个变量
        DATA_PATH = r"D:\Desktop\files\huawei\repo\continual_learning\TRACE\LLM-CL_Benchmark"
        MODEL_PATH = r"D:\Desktop\files\huawei\repo\continual_learning\TRACE\Qwen-0.6B"
        OUTPUT_DIR = r"./outputs_debug"

        sys.argv.extend([
            "--data_path", DATA_PATH,
            "--dataset_name", "C-STANCE,FOMC",  # 调试时数据少一点，跑得快
            "--model_name_or_path", MODEL_PATH,
            "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "1",
            "--gradient_accumulation_steps", "1",
            "--max_prompt_len", "64",    # ⚡ 调小长度，Debug 启动更快
            "--max_ans_len", "64",       # ⚡ 调小长度
            "--learning_rate", "1e-5",
            "--num_train_epochs", "1,1",
            "--seed", "42",
            "--zero_stage", "2",
            "--deepspeed",               # 必须保留
            "--print_loss",
            "--CL_method", "lora",
            "--output_dir", OUTPUT_DIR,
            "--local_rank", "0",          # 显式告诉代码我是 rank 0
            
            "--num_sinks", "4",        # 脚本里是 128 (默认是0，这个很重要)
            "--use_sink", "True",
            "--sliding_window", "512",
            "--segment_size", "512"     # 脚本里是 2048 (默认也是2048)
            # ===================================
        ])
    
    # ===========================================================

    main()