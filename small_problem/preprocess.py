from data import Universe
from sklearn.model_selection import train_test_split
import tensorflow as tf

def preprocess(num, batch_size):
    universe = Universe()
    data = universe.gen_dataset(num)
    inputs = []
    labels = []
    for (prob, sol, std, ocrr) in data:
        inputs.append(ocrr)
        if sol == std:
            labels.append([0, 1])
        else:
            labels.append([1, 0])
    inputs_train, inputs_val, labels_train, labels_val = train_test_split(inputs, labels, test_size=0.1)
    dataset_train = tf.data.Dataset.from_tensor_slices((inputs_train,
                                                        labels_train)).shuffle(num).batch(batch_size,
                                                                                          drop_remainder=True)
    dataset_val = tf.data.Dataset.from_tensor_slices((inputs_val,
                                                      labels_val)).shuffle(num).batch(batch_size,
                                                                                      drop_remainder=True)

    return universe, dataset_train, dataset_val