import tensorflow as tf


def accuracy(pred_id, label):
    return tf.reduce_mean(tf.cast(pred_id == label, tf.float32))
