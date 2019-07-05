import numpy as np
import tensorflow as tf
from tensorflow import keras
import data_gen2 as dg

length = 16
num_units = 64
batch_size = 100
input_size = 26
label_size = 2
hidden_units = 40
num_epochs=2000

inputs = keras.Input(shape=(length, input_size), dtype=tf.float32)
lstm1 = keras.layers.LSTM(num_units, input_shape=(None, length, input_size), return_sequences=True)(inputs)
lstm2 = keras.layers.LSTM(num_units, return_sequences=False, dropout=0.2, recurrent_dropout=0.1)(lstm1)
relu = keras.layers.Dense(hidden_units, activation=tf.nn.relu)(lstm2)
out = keras.layers.Dense(label_size, activation=tf.nn.softmax)(relu)

model = keras.Model(inputs=inputs, outputs=out)

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='categorical_crossentropy', metrics=['accuracy'])

for i in range(num_epochs):
    data, label = dg.gen_data_batch(batch_size, length)
    if (i % 100 == 0):
        print("\ncurrent_accuracy at epoch ", i)
        pos_data, pos_label = dg.gen_data_batch(100, length, True)
        neg_data, neg_label = dg.gen_data_batch(100, length, False)
        loss1, accuracy1 = model.evaluate(pos_data, pos_label, batch_size=batch_size)
        loss2, accuracy2 = model.evaluate(neg_data, neg_label, batch_size=batch_size)
        print("\nPositive Loss: %.3f\n Positive Acc: %.3f" % (loss1, accuracy1))
        print("\nNegative Loss: %.3f\n Negative Acc: %.3f" % (loss2, accuracy2))
    model.fit(data, label, epochs=1, batch_size=batch_size)
model.save('my_model.h5')


