import tensorflow as tf
from accuracy import accuracy
from evaluate import evaluate


def train(dataset_train, dataset_val, num_batch, epochs, model, optimizer, loss_function, batch_size):
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for batch, (inputs, label) in enumerate(dataset_train):
            with tf.GradientTape() as tape:
                pred = model(inputs)
                loss = loss_function(label, pred)
                total_loss += loss
                pred_id = tf.argmax(pred, axis=1)
                acc = accuracy(pred_id, label)
                total_acc += acc
                variables = model.variables
                grad = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(grad, variables))
        print("Epoch: {}/{}, Loss: {}, Accuracy: {}".format(epoch, epochs,
                                                            total_loss / num_batch, total_acc / num_batch))
        evaluate(dataset_val, model, num_batch, batch_size)
