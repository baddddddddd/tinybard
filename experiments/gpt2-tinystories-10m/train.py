import torchinfo
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

from tinyfriend.tokenizers import TinyStoriesBpe8kTokenizer

tokenizer = TinyStoriesBpe8kTokenizer().tokenizer

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_embd=384,
    n_layer=4,
    n_head=6,
    bos_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)
torchinfo.summary(model)


def tokenize(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    block_size = 256
    concatenated = [token for input_ids in examples["input_ids"] for token in input_ids]
    total_length = (len(concatenated) // block_size) * block_size
    input_ids = [
        concatenated[i : i + block_size] for i in range(0, total_length, block_size)
    ]
    return {"input_ids": input_ids, "labels": input_ids.copy()}


dataset = load_dataset("roneneldan/TinyStories")
tokenized_dataset = dataset.map(
    tokenize, batched=True, remove_columns=["text"], num_proc=2
)
lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=2)

training_args = TrainingArguments(
    output_dir="./checkpoints/gpt2-tinystories-10m",
    learning_rate=5e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_steps=20000,
    weight_decay=0.01,
    fp16=True,
    warmup_steps=500,
    eval_strategy="epoch",
    eval_steps=500,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    logging_dir="./logs/gpt2-tinystories-10m",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    tokenizer=tokenizer,
)

trainer.train(resume_from_checkpoint=True)
