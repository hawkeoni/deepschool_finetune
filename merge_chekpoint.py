import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1]
lora_path = sys.argv[2]


model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.half)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
model.save_pretrained("model_with_lora")
tokenizer.save_pretrained("model_with_lora")