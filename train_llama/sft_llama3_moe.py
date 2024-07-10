import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch

save_ckpt_path="[path of saving trained moe model]"

# 将JSON文件转换为CSV文件
df = pd.read_json('data/llama3_train_origin_v1.json')
ds = Dataset.from_pandas(df)
print(ds[:3])

from mergoo.models.modeling_llama import LlamaForCausalLM


model_path = "[path to initial moe model]"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
)
model.enable_input_require_grads() 


n_weights, n_router_weights  = 0,0
for name, weight in model.named_parameters():
    if "gate" not in name:
        weight.requires_grad_(False)
        n_router_weights += 1
    n_weights += 1


def process_func(example):
    MAX_LENGTH = 512    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)


args = TrainingArguments(
    output_dir=save_ckpt_path,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    logging_steps=5,
    num_train_epochs=1,
    save_steps=200,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
