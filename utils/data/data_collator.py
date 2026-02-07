# D:\Desktop\files\huawei\repo\continual_learning\TRACE\utils\data\data_collator.py
import logging
import torch
from transformers.data.data_collator import dataclass,Optional,Any,Union,PreTrainedTokenizerBase,PaddingStrategy
from inference.ICL import TASK_PROMT, Constrained_PROMPT

logger = logging.getLogger(__name__)

# 全局开关
testMode = True

@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True  # ‘longest’
    max_prompt_len: Optional[int] = None
    max_ans_len: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 1
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    inference: bool = False
    demonstrations: Optional[Any] = None
    task: str = None

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        model_inputs = self.decoder_call(batch, self.return_tensors)

        return model_inputs

    # only support left padding for now
    def tokenize(self, sentence, cutoff_len, add_bos_token=True, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            sentence,
            truncation=True,
            max_length=cutoff_len,
            add_special_tokens=False,
            padding=False,
            return_tensors=None,
        )

        if (
                len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if (
                len(result["input_ids"]) < cutoff_len
                and add_bos_token
        ):
            result["input_ids"] = [self.tokenizer.bos_token_id] + result["input_ids"]
            result["attention_mask"] = [1] + result["attention_mask"]

        result["labels"] = result["input_ids"].copy()

        return result

    # support decoder-only models for left padding
    def decoder_call(self, batch, return_tensors):
        # ================= [STEP MONITORING CONFIG] =================
        # 请确保这里的 gradient_accumulation_steps 与你的脚本一致 (16)
        GRAD_ACCUM_STEPS = 16  
        # 你想要监控的 Global Steps
        TARGET_GLOBAL_STEPS = [14, 15, 16, 17]
        
        # 初始化计数器 (如果不存在)
        if not hasattr(self, 'total_micro_step_count'):
            self.total_micro_step_count = 0
            
        # 计算当前的 Global Step 和 Accumulation Step
        current_global_step = self.total_micro_step_count // GRAD_ACCUM_STEPS
        current_accum_step = self.total_micro_step_count % GRAD_ACCUM_STEPS
        
        # 判断是否需要打印
        should_print = testMode and (current_global_step in TARGET_GLOBAL_STEPS)
        # ============================================================

        # to fix the bug
        sources = []
        gts = []
        tokenized_sources = []
        label_lens = []  # 用于存储每个label的长度
        actual_max_len = 0  # 用于存储batch中的实际最大长度
        limit_len = self.max_prompt_len + self.max_ans_len if not self.inference else self.max_prompt_len

        # 使用 enumerate 以便我们能识别出 batch 中的第一个样本进行打印
        for i, instance in enumerate(batch):
            instruction = instance['prompt']
            label = instance['answer']
            sources.append(instruction)
            gts.append(label)
            
            if should_print:
                # ================= [DEBUG START] =================
                # 仅在 Rank 0 (主进程) 且是 Batch 的第一个样本时打印，避免刷屏
                if i == 0:
                    is_main_process = True
                    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                        is_main_process = False
                    
                    if is_main_process:
                        print(f"\n{'='*10} [Global Step {current_global_step} | Accum {current_accum_step}/{GRAD_ACCUM_STEPS}] {'='*10}")
                        # 1. 打印 Keys
                        print(f"Sample Keys: {list(instance.keys())}")
                        
                        # 2. 单词数 (基于空格粗略计算)
                        prompt_words = len(instruction.split())
                        ans_words = len(label.split())
                        print(f"Word Count -> Prompt: {prompt_words}, Answer: {ans_words}")
                # =================================================

            if not self.inference:
                tokenized_label = self.tokenize(label, limit_len, add_bos_token=False, add_eos_token=True)
                tokenize_source = self.tokenize(instruction + label, limit_len, add_bos_token=True, add_eos_token=True)
                label_lens.append(len(tokenized_label["input_ids"]))
                tokenized_sources.append(tokenize_source)
                
                if should_print:
                    # ================= [DEBUG INFO Continue] =================
                    if i == 0 and is_main_process:
                        # 3. 打印真实的 Token 数 (Padding前)
                        src_len = len(tokenize_source['input_ids'])
                        lbl_len = len(tokenized_label['input_ids'])
                        print(f"Token Count -> Total: {src_len} (Limit: {limit_len}), Label portion: {lbl_len}")
                        
                        # 警告：检查是否达到长度限制
                        if src_len >= limit_len:
                            print(f"⚠️ [WARNING] Sample reached MAX LENGTH ({limit_len})! Possible truncation.")

                        # 4. 打印截断预览 (前100个字符，替换换行符以免排版混乱)
                        safe_prompt = instruction.replace('\n', ' ')
                        safe_label = label.replace('\n', ' ')
                        
                        p_cut = 100
                        print(f"Prompt (Top {p_cut}): {safe_prompt[:p_cut]}...")
                        print(f"Answer (Top {p_cut}): {safe_label[:p_cut]}...")
                        print(f"{'='*60}\n")
                    # =========================================================

            else:
                if self.demonstrations!=None:
                    task_prompt = ""
                    task_prompt += TASK_PROMT[self.task]
                    if self.task!="MeetingBank": #MeetingBank不给例子
                        task_prompt += Constrained_PROMPT
                    for demonstration in self.demonstrations:
                        if self.task=="Py150":
                            task_prompt+= "Code:\n"
                        task_prompt+=demonstration["prompt"]
                        task_prompt+=demonstration["answer"]+"\n\n"
                    
                    if self.task=="Py150":
                        task_prompt+= "Code:\n"
                    # task_prompt += Constrained_PROMPT
                    if self.task!="Py150": #Py150不带prompt
                        instruction = instruction[len(TASK_PROMT[self.task]):]
                    instruction = task_prompt+instruction
                tokenize_source = self.tokenize(instruction, limit_len, add_bos_token=True, add_eos_token=False)
                tokenized_sources.append(tokenize_source)

                if should_print:
                    # ================= [DEBUG INFO Inference] =================
                    if i == 0 and is_main_process:
                        print(f"Token Count -> Input: {len(tokenize_source['input_ids'])}")
                        safe_prompt = instruction.replace('\n', ' ')
                        print(f"Prompt (Top 50): {safe_prompt[:50]}...")
                        print(f"{'='*60}\n")
                    # ==========================================================

            if len(tokenize_source["input_ids"]) > actual_max_len:
                actual_max_len = len(tokenize_source["input_ids"])
        
        # ================= [COUNTER INCREMENT] =================
        # 非常重要：每次调用必须增加计数器
        self.total_micro_step_count += 1
        # =======================================================

        # 取batch中的最大长度和limit_input_len中的最小值作为实际padding长度
        # 并确保长度是pad_to_multiple_of的倍数
        actual_pad_len = (
                    (actual_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of)

        # 对于left padding和prompt部分的mask
        for idx in range(len(tokenized_sources)):
            pad_len = actual_pad_len - len(tokenized_sources[idx]["input_ids"])
            assert sum(tokenized_sources[idx]["attention_mask"]) == len(tokenized_sources[idx]["input_ids"])
            tokenized_sources[idx]["input_ids"] = [self.tokenizer.pad_token_id] * pad_len + tokenized_sources[idx][
                "input_ids"]

            tokenized_sources[idx]["attention_mask"] = [0] * pad_len + tokenized_sources[idx]["attention_mask"]

            if not self.inference:
                label_len = label_lens[idx]
                label_mask_len = actual_pad_len - label_len
                tokenized_sources[idx]["labels"] = [-100] * label_mask_len + tokenized_sources[idx]["labels"][
                                                                             -label_len:]
                assert len(tokenized_sources[idx]["input_ids"]) == len(tokenized_sources[idx]["attention_mask"]) == len(
                    tokenized_sources[idx]["labels"]) == actual_pad_len

        model_inputs = {'input_ids': torch.tensor([source["input_ids"] for source in tokenized_sources]),
                        'attention_mask': torch.tensor([source["attention_mask"] for source in tokenized_sources])}

        if not self.inference:
            model_inputs['labels'] = torch.tensor([source["labels"] for source in tokenized_sources])

        model_inputs['sources'] = sources
        if self.inference:
            model_inputs['gts'] = gts

        return model_inputs
