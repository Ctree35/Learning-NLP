from tensorflow import keras

class Model(keras.Model):
    def __init__(self, embedding_size, hidden_unit, vocab_size):
        super(Model, self).__init__()
        self.embed = keras.layers.Embedding(vocab_size, embedding_size)
        self.linear = keras.layers.Dense(hidden_unit, activation='relu')
        self.final = keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.embed(inputs)
        x = self.linear(x)
        x = self.final(x)
        return x