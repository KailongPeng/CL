# D:\Desktop\files\huawei\repo\continual_learning\TRACE\model\base_model.py
import torch
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import os
import time
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds # to be continued
from transformers import GenerationConfig
generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)


class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_task_list,
                 eval_task_list,
                 test_task_list,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_task_list = train_task_list
        self.eval_task_list = eval_task_list
        self.test_task_list = test_task_list
        self.args = args
        
        
    def perplexity_evaluation(self, eval_dataloader, device):
        # 验证集上测困惑度
        self.model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = self.model(**batch, use_cache=False)
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


    def train_one_task(self, task, i_task, epochs):
        # 在单独某个任务上训练
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
        
        #### TRAIN ####
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss

                testMode = True
                if testMode:
                    # ======================= [Start] 新增修改部分 =======================
                    # 仅在主进程打印，且为了不严重拖慢训练，可以加上 if step % 10 == 0: 的判断
                    # 但既然你要求“每一个step”，这里就不加频率限制了
                    if self.args.global_rank == 0:
                        # 1. 获取 Logits: [batch_size, seq_len, vocab_size]
                        logits = outputs.logits
                        # 2. 获取预测的 Token ID (Argmax): [batch_size, seq_len]
                        pred_ids = torch.argmax(logits, dim=-1)

                        # 3. 取 Batch 中的第一条数据 (索引 0)
                        input_seq_ids = batch['input_ids'][0]
                        pred_seq_ids = pred_ids[0]

                        # 4. 解码成字符串
                        # input_text: 实际喂给模型的输入
                        # pred_text: 模型在这个位置预测输出的下一个 token 组成的序列
                        input_text = self.tokenizer.decode(input_seq_ids, skip_special_tokens=True)
                        pred_text = self.tokenizer.decode(pred_seq_ids, skip_special_tokens=True)

                        # 5. 打印对比
                        print(f"\n{'='*20} Step {step} Monitor {'='*20}")
                        print(f"\n[Input Text]:\n{input_text[:200]}...") # 只打印前200字符，避免刷屏太长
                        print(f"\n[Model Pred]:\n{pred_text[:200]}...") # 这里的 Pred 是模型根据当前字预测的“下一个字”
                        print(f"\n{'='*55}\n")
                    # ======================= [End] 新增修改部分 =======================

                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()
    
    
    def train_continual(self):
        for i_task, task in enumerate(self.train_task_list):
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            self.save_model(i_task)

    
    def save_model(self, round):
        if self.args.output_dir is not None:
            print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)

        if self.args.global_rank == 0:
            save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(round))

        if self.args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(self.model,
                                  self.args.global_rank,
                                  self.args.output_dir,
                                  zero_stage=self.args.zero_stage,
                                  sub_folder=str(round))
        print_rank_0('Sucessful saving model after round {}'.format(round), self.args.global_rank)
        
