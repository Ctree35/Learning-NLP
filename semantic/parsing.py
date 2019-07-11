import tensorflow as tf
from sklearn.model_selection import train_test_split
import re
import numpy as np
import os

tf.enable_eager_execution()

DATA_PATH = './geo880/geo880.txt'


def preprocess_quest(w):
    w = w.lower().rstrip('\n')
    # Delete useless symbols
    w = re.sub(r'parse\(\[', '', w)
    w = re.sub(r',\?]', '', w)
    w = re.sub(r",'\.']", '', w)
    # Deal with typos
    w = re.sub(r'\bhamsphire\b', 'hampshire', w)
    w = re.sub(r'\bmississsippi\b', 'mississippi', w)
    w = re.sub(r'\bcites\b', 'cities', w)
    # Separate by whitespaces
    w = re.sub(r',', ' ', w)
    w = re.sub(r'\s+', ' ', w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    # print(w)
    return w

def preprocess_answer(w):
    w = w.lower().rstrip('\n')
    # Delete useless symbols
    w = re.sub(r'answer\(a,', '', w)
    w = re.sub(r'\)\)\.', '', w)
    # w = re.sub(r"'", '', w)
    # Deal with typos
    w = re.sub(r'\bhamsphire\b', 'hampshire', w)
    w = re.sub(r'\bmississsippi\b', 'mississippi', w)
    w = re.sub(r'\bcites\b', 'cities', w)
    # Separate by whitespaces
    w = re.sub(r',', ' , ', w)
    w = re.sub(r'\(', ' ( ', w)
    w = re.sub(r'\)', ' ) ', w)
    w = re.sub(r'\\\+', ' \\+ ', w)
    w = re.sub(r"'", " ' ", w)
    w = re.sub(r'\s+', ' ', w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    # print(w)
    return w

def create_dataset(path):
    lines = open(path,'r').readlines()
    pairs = list()
    for l in lines:
        que, par = l.split(', ')
        pairs.append([preprocess_quest(que), preprocess_answer(par)])
    return pairs


print(create_dataset(DATA_PATH))

class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def load_dataset(path):
    pairs = create_dataset(path)
    inp_lang = LanguageIndex(que for que, par in pairs)
    targ_lang = LanguageIndex(par for que, par in pairs)
    input_tensor = [[inp_lang.word2idx[s] for s in que.split(' ')] for que, par in pairs]
    target_tensor = [[targ_lang.word2idx[s] for s in par.split(' ')] for que, par in pairs]

    def max_length(tensor):
        return max(len(t) for t in tensor)
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen=max_length_tar,
                                                                  padding='post')

    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar


input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(DATA_PATH)

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1)

# Config
is_training = False
is_eval = True
buffer_size=len(input_tensor_train)
batch_size=64
num_batch=buffer_size // batch_size
embedding_dim=256
units=1024
vocab_inp_size=len(inp_lang.word2idx)
vocab_tar_size=len(targ_lang.word2idx)
dataset=tf.data.Dataset.from_tensor_slices((input_tensor_train,target_tensor_train)).shuffle(buffer_size).batch(batch_size,drop_remainder=True)


# Model
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True))

    def call(self, x):
        x = self.embedding(x)
        out, state_fw, state_bw = self.gru(x)
        return out, state_fw, state_bw


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.nn.relu)

        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_out):
        hidden_expand = tf.expand_dims(hidden, axis=1)
        score = self.V(tf.nn.tanh(self.W1(enc_out) + self.W2(hidden_expand)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * enc_out
        context_vector = tf.reduce_sum(context_vector, axis=1)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=2)
        out, state = self.gru(x)
        out = tf.reshape(out, (-1, self.dec_units))
        x = self.fc(out)
        return x, state


encoder=Encoder(vocab_inp_size,embedding_dim,units,batch_size)
decoder=Decoder(vocab_tar_size,embedding_dim,2 * units,batch_size)
optimizer = tf.train.AdamOptimizer()


def loss_function(real, pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real,
                                                          logits=pred)
    return tf.reduce_mean(loss)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Train
if is_training:
    epochs = 10

    for epoch in range(epochs):
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            with tf.GradientTape() as tape:
                enc_out, enc_hidden_fw, enc_hidden_bw = encoder(inp)
                dec_hidden = tf.concat([enc_hidden_fw, enc_hidden_bw], axis=-1)
                dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * batch_size, 1)

                for t in range(1, max_length_targ):
                    pred, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
                    loss += loss_function(targ[:, t], pred)
                    dec_input = tf.expand_dims(tf.argmax(pred, axis=1), 1)
                batch_loss = loss / max_length_targ
                total_loss += batch_loss
                variables = encoder.variables + decoder.variables
                grad = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(grad, variables))
        checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / num_batch))


# Evaluation
def evaluate(question, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    question = tf.expand_dims(question, axis=0)
    enc_out, enc_hidden_fw, enc_hidden_bw = encoder(question)
    enc_hidden = tf.concat([enc_hidden_fw, enc_hidden_bw], axis=-1)

    # Beam Search
    beam_size = 5
    sequences = [[list(), 0, enc_hidden]]
    sequences[0][0] += [targ_lang.word2idx['<start>']]
    final = [[list(), 0, enc_hidden]]
    for t in range(max_length_targ):
        if beam_size <= 0:
            break
        all_candidates = list()
        min_score = 1e10
        min_local_score = 1e10
        for i in range(len(sequences)):
            seq, score, dec_hidden = sequences[i]
            if seq[-1] == targ_lang.word2idx['<end>']:
                final.append(sequences[i])
                beam_size -= 1
                continue
            dec_input = tf.expand_dims([seq[-1]], 1)
            pred, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
            for j in range(vocab_tar_size):
                local_score = -np.log(pred[0][j].numpy()+1e-12)
                new_score = score + local_score
                if new_score > 3 * min_score:
                    continue
                if local_score > 3 * min_local_score:
                    continue
                if new_score < min_score:
                    min_score = new_score
                if local_score < min_local_score:
                    min_local_score = local_score
                candidate = [seq+[j], new_score, dec_hidden]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_size]
    final = sorted(final, key=lambda x: x[1])
    return final[0][0]


def accuracy(pred, label):
    if np.all(pred == label):
        return 1
    return 0

if is_eval:
    buf_siz = len(input_tensor_val)
    total_acc = 0
    for i in range(buf_siz):
        predict = evaluate(input_tensor_val[i], encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        while len(predict) < max_length_targ:
            predict.append(0)
        acc = accuracy(predict, target_tensor_val[i])
        total_acc += acc
    total_acc = total_acc / buf_siz
    print("Total Accuracy is: {:.4f}".format(total_acc))
