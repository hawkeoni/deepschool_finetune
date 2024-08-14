import sys
from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

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
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset_preproc_with_tokenizer = partial(dataset_preproc, tokenizer=tokenizer)

# Добавляем в датасет колонку text, на которой и будем учиться
dataset = load_dataset("json", data_files={"train": "mailru_qa_dataset.jsonl"})
dataset = dataset.map(dataset_preproc_with_tokenizer)

# Определяем параметры обучения
sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=300,
    output_dir="./models",
)
trainer = SFTTrainer(
    model_name,
    train_dataset=dataset,
    args=sft_config,
)
# Вызываем обучение
trainer.train()