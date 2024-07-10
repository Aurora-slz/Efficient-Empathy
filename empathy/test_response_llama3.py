import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto
import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from mergoo.models.modeling_llama import LlamaForCausalLM



load_path = "[path to trained moe model]"
save_response_path = "./response.txt"
load_data_path = "/data/test_response.json"

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    load_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
)


import json

def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        print(data[0])
    return data

load_path = load_data_path
test_data = load_file(load_path)

with open(save_response_path, 'w', encoding='utf-8') as f1:
        
    for i in range(0, len(test_data)):
        # messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages = []
        messages = messages + test_data[i]["conversations"][:-1]
        # print(messages)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            eos_token_id=tokenizer.encode('<|eot_id|>')[0]
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)

        f1.write('Context:'+str(test_data[i]["conversations"])+'\n')
        f1.write('Greedy:'+response+'\n')
        f1.write('Ref:'+test_data[i]["conversations"][-1]["content"]+'\n')
        split = '-'*10
        f1.write(split+'\n')


        