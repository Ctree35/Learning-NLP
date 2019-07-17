from accuracy import accuracy
import tensorflow as tf


def evaluate(dataset, model, num_batch, batch_size):
    total_acc = 0
    for (inputs, label) in dataset:
        pred = model(inputs)
        pred_id = tf.argmax(pred, axis=1)
        acc = accuracy(pred_id, label)
        total_acc += acc
    print("Accuracy On Validation Set: {}".format(total_acc / num_batch))