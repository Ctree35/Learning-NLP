import tensorflow as tf


def loss_function(label, pred):
    loss = tf.keras.losses.categorical_crossentropy(label, pred)
    return tf.reduce_mean(loss)
