# D:\Desktop\files\huawei\repo\continual_learning\TRACE\utils\model\model_utils.py
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
from transformers.integrations import HfDeepSpeedConfig
from transformers import LlamaForCausalLM, LlamaConfig


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    args=None
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    if disable_dropout:
        model_config.dropout = 0.0

    if args is not None:
        if model_config.model_type == "memorized_qwen": # 避免了对于qwen也注入sliding window和segment_size
            # 注入 num_sinks
            if hasattr(args, "num_sinks"):
                model_config.num_sinks = args.num_sinks
            
            # 注入 use_sink (注意处理字符串 "False"/"True" 转布尔值)
            if hasattr(args, "use_sink"):
                # 如果传入的是字符串，转为 bool；如果是 bool 则直接用
                if isinstance(args.use_sink, str):
                    model_config.use_attn_sink = (args.use_sink.lower() == 'true')
                else:
                    model_config.use_attn_sink = bool(args.use_sink)
    
            # 注入 sliding_window
            if hasattr(args, "sliding_window"):
                model_config.sliding_window = args.sliding_window
                
            # 注入 segment_size
            if hasattr(args, "segment_size"):
                model_config.segment_size = args.segment_size
                
        print(f"Injecting config: use_attn_sink={getattr(model_config, 'use_attn_sink', 'N/A')}, num_sinks={getattr(model_config, 'num_sinks', 'N/A')}")
    # =============================================================

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # <--- 在这里添加这一行
        torch_dtype=torch.bfloat16)

    # llama use eos_token_id but not end_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    # compatible with OPT and llama2
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model
