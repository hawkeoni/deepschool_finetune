import sys
from functools import partial

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

def dataset_preproc(tokenizer, sample):
    messages = [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]}
    ]
    return {"text": tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        )
    }

# Создаем датасет и определяем функцию, которая объединит вопрос и ответ в нужном формате
model_name = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset_preproc_with_tokenizer = partial(dataset_preproc, tokenizer=tokenizer)

# Добавляем в датасет колонку text, на которой и будем учиться
dataset = load_dataset("json", data_files={"train": "mailru_qa_dataset.jsonl"})
dataset = dataset.map(lambda x: dataset_preproc_with_tokenizer(sample=x))

# Учить будем только Lora добавку ко всем линейным слоям
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

# Определяем параметры обучения
sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=300,
    output_dir="./training_checkpoints",
    logging_strategy="steps",
    logging_steps=10,
    num_train_epochs=1,
    learning_rate=2e-5,
    gradient_accumulation_steps=4,
    per_gpu_train_batch_size=8,
    save_strategy="steps",
    save_steps=100,
    gradient_checkpointing=True,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=None,
    args=sft_config,
    peft_config=peft_config,
)
print("Started training")
trainer.train()
print("Finished training")