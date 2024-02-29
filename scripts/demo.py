import pandas as pd
from tqdm import tqdm
import json
import torch
from transformers import GenerationConfig, LlamaTokenizer
import os
import sys
sys.path.append('/home/weiguang/Mixture-of-Domain-Adapters/')
import pytorch_lightning as pl ## have to import first to let torch has .fx
from src.models.transformers import LlamaForCausalLM, LlamaConfig
from scripts.llama_run_qa import Model
from peft import PeftModel, PeftConfig, LoraConfig, TaskType


# load the original LlaMa
# MODEL_NAME = "/data/weiguang/Llama2-Chinese-7b-Chat"
# tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
# model = LlamaForCausalLM.from_pretrained(MODEL_NAME).to('cuda:2')

# load our fine-tuned Llama model
best_model = Model.load_from_checkpoint('/home/weiguang/Mixture-of-Domain-Adapters/results/cls_finetune_mixda_seed42_bos_notn/epoch=4-step=1245000.ckpt', 
                                        layers = '7,11', load_adapters = 'llama_translated_7_11/model20231225-114908.pt6,llama_translated_7_11/model20231225-115001.pt6',
                                        adapter_type = 'lora')
# type(best_model.model): <class 'peft.peft_model.PeftModelForCausalLM'>
# type(best_model.model.base_model): <class 'peft.tuners.lora.model.LoraModel'>
# type(best_model.model.base_model.model): <class 'src.models.transformers.modeling_llama.LlamaForCausalLM'>
tokenizer = best_model.tokenizer
model = best_model.model.to('cuda:0')
model = model.merge_and_unload() ## merge the lora adapters with base model to use it as a standalone base model

# config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=8, lora_dropout=0.1)
# model = PeftModel.from_pretrained(model, peft_model_id).to("cuda:0")
generation_config = GenerationConfig(max_new_tokens=512)
model.eval()

tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token


while True:
    text = input()
    if text == 'quit':
        break
    text = '问题: '+text+' 回答: '
    encoding = tokenizer(text, padding=True, max_length=512, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], generation_config=generation_config)
    answer = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    print(answer)
    

# batch_size = 8
# for i in tqdm(range(start_index+1, len(data), batch_size)):
#     prompt = []
#     for j in range(batch_size):
#         if i+j < len(data):
#             prompt.append(f"<s>问题: "+data['question'][i+j] + data['description'][i+j]+"\n回答: ")
#     encoding = tokenizer(prompt, padding=True, max_length=512, truncation=True, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], generation_config=generation_config)
#     answer = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
#     with open(path, "a", encoding='utf8') as f:
#         for j in range(batch_size):
#             if i+j < len(data):
#                 json.dump({i+j: answer[j]}, f, ensure_ascii=False)
#                 f.write('\n')  # Add a newline to separate entries