class Config:
    def __init__(self):
        self.embedding_size = 256
        self.hidden_size = 64
        self.batch_size = 64
        self.epochs = 100
        self.data_size = 1000
        self.vocab_size = 100
        self.num_batch = self.data_size // self.batch_size
