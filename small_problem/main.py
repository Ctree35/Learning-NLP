from preprocess import preprocess
from model import Model
from loss_function import loss_function
from train import train
from config import Config
from tensorflow import keras

config = Config()
universe, dataset_train, dataset_val = preprocess(config.data_size, config.batch_size)
model = Model(config.embedding_size, config.hidden_size, config.vocab_size)
optimizer = keras.optimizers.Adam()
train(dataset_train, dataset_val, config.num_batch, config.epochs, model, optimizer, loss_function, config.batch_size)
