class BaseTrainer:
    def train(self, resume_from_checkpoint: str | bool = False):
        raise NotImplementedError("train() method is not implemented")
