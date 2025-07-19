class TextGenStreamer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def put(self, token_ids):
        decoded = self.tokenizer.decode(token_ids)
        print(decoded, end="", flush=True)

    def end(self):
        print()
