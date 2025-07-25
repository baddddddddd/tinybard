import torchinfo

from tinyfriend.datasets import TinyStoriesDataset
from tinyfriend.models import TinyFriendTransformerConfig, TinyFriendTransformerModel
from tinyfriend.tokenizers import StrippedAsciiTokenizer
from tinyfriend.trainer import CausalLmTrainer, TrainingConfig


seq_len = 256

tokenizer = StrippedAsciiTokenizer(
    max_length=seq_len, padding=True, return_overflowing_tokens=True
)

model_config = TinyFriendTransformerConfig(
    vocab_size=tokenizer.get_vocab_size(),
    max_len=seq_len,
    embed_dim=256,
    num_layers=6,
    num_heads=8,
    dim_feedforward=512,
    dropout=0.1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

training_config = TrainingConfig(
    output_dir="./checkpoints/tinyfriend-transformer-3m-char-tinystories",
    num_train_epochs=10,
    learning_rate=1e-3,
    betas=(0.9, 0.98),
    weight_decay=1e-3,
    label_smoothing=0.0,
    train_batch_size=64,
    save_steps=100,
)

model = TinyFriendTransformerModel(model_config)
torchinfo.summary(model, depth=4)

dataset = TinyStoriesDataset(split="train[:100000]", tokenizer=tokenizer)

trainer = CausalLmTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_config,
    train_dataset=dataset,
)

trainer.train(resume_from_checkpoint=True)
