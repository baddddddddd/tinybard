import argparse

import torchinfo

from tinyfriend.models.tinyfriend_transformer import (
    TinyFriendTransformerConfig,
    TinyFriendTransformerModel,
)
from tinyfriend.utils import TextGenStdoutStreamer
from tinyfriend.tokenizers import TinyStoriesBpe8kTokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the checkpoint folder"
)

args = parser.parse_args()

tokenizer = TinyStoriesBpe8kTokenizer()
model = TinyFriendTransformerModel.from_pretrained(args.checkpoint)

streamer = TextGenStdoutStreamer(tokenizer)

torchinfo.summary(model)
while True:
    prompt = input("\n>>> ")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0, :-1]
    output = model.generate(input_ids, streamer=streamer)
