#!/usr/bin/env python
# coding: utf-8

# In[17]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import time


# In[18]:


tf.enable_eager_execution()


# In[19]:


path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip', 
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"


# In[20]:


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s)
                  if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    return word_pairs


# In[21]:


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


# In[22]:


def max_length(tensor):
    return max(len(t) for t in tensor)

def load_dataset(path, num_examples):
    pairs = create_dataset(path, num_examples) 
    inp_lang = LanguageIndex(sp for en, sp in pairs)
    targ_lang = LanguageIndex(en for en, sp in pairs)
    input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]
    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_length_tar, 
                                                                  padding='post')
    
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar


# In[23]:


num_examples=30000
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file, num_examples)


# In[24]:


input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)


# In[25]:


len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)


# In[26]:


buffer_size=len(input_tensor_train)
batch_size=64
num_batch=buffer_size // batch_size
embedding_dim=256
units=1024
vocab_inp_size=len(inp_lang.word2idx)
vocab_tar_size=len(targ_lang.word2idx)
dataset=tf.data.Dataset.from_tensor_slices((input_tensor_train,target_tensor_train)).shuffle(buffer_size).batch(batch_size,drop_remainder=True)


# In[27]:


class Encoder(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dim,enc_units,batch_sz):
        super(Encoder,self).__init__()
        self.batch_sz=batch_sz
        self.enc_units=enc_units
        self.embedding=tf.keras.layers.Embedding(vocab_size,embedding_dim)
        self.gru=tf.keras.layers.GRU(enc_units,return_sequences=True,return_state=True)
    
    def call(self,x,hidden):
        x=self.embedding(x)
        out,state=self.gru(x, initial_state=hidden)
        return out,state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


# In[28]:


class Decoder(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dim,dec_units,batch_sz):
        super(Decoder,self).__init__()
        self.batch_sz=batch_sz
        self.dec_units=dec_units
        self.embedding=tf.keras.layers.Embedding(vocab_size,embedding_dim)
        self.gru=tf.keras.layers.GRU(dec_units,return_sequences=True,return_state=True)
        self.fc=tf.keras.layers.Dense(vocab_size,activation=tf.nn.relu)
        
        self.W1=tf.keras.layers.Dense(self.dec_units)
        self.W2=tf.keras.layers.Dense(self.dec_units)
        self.V=tf.keras.layers.Dense(1)
        
    def call(self,x,hidden,enc_out):
        hidden_expand=tf.expand_dims(hidden, axis=1)
        score=self.V(tf.nn.tanh(self.W1(enc_out)+self.W2(hidden_expand)))
        attention_weights=tf.nn.softmax(score, axis=1)
        context_vector=attention_weights * enc_out
        context_vector=tf.reduce_sum(context_vector,axis=1)
        x=self.embedding(x)
        x=tf.concat([tf.expand_dims(context_vector,axis=1),x],axis=2)
        out,state=self.gru(x)
        out=tf.reshape(out,(-1,self.dec_units))
        x=self.fc(out)
        return x,state,attention_weights
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


# In[29]:


encoder=Encoder(vocab_inp_size,embedding_dim,units,batch_size)
decoder=Decoder(vocab_tar_size,embedding_dim,units,batch_size)


# In[30]:


optimizer=tf.train.AdamOptimizer()

def loss_function(real,pred):
    mask=1-np.equal(real,0)
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real,
                                                        logits=pred)*mask
    return tf.reduce_mean(loss)


# In[31]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# In[32]:


epochs=10

for epoch in range(epochs):
    hidden=encoder.initialize_hidden_state()
    total_loss=0
    for (batch,(inp,targ)) in enumerate(dataset):
        loss=0
        with tf.GradientTape() as tape:
            enc_out,enc_hidden=encoder(inp,hidden)
            dec_hidden=enc_hidden
            dec_input=tf.expand_dims([targ_lang.word2idx['<start>']]*batch_size,1)
            
            for t in range(1,max_length_targ):
                pred, dec_hidden,_=decoder(dec_input,dec_hidden,enc_out)
                loss+=loss_function(targ[:,t],pred)
                dec_input=tf.expand_dims(tf.argmax(pred,axis=1),1)
            batch_loss=loss/max_length_targ
            total_loss+=batch_loss
            variables=encoder.variables+decoder.variables
            grad=tape.gradient(loss,variables)
            optimizer.apply_gradients(zip(grad,variables))
            
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / num_batch))


# In[ ]:


def evaluate(sentene,encoder,decoder,inp_lang,targ_lang,max_length_inp,max_length_targ):
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result=''
    
    hidden = encoder.initialize_hidden_state()
    enc_out, enc_hidden=encoder(inputs,hidden)
    dec_hidden=enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 1)
    
    for t in range(max_length_targ):
        pred, dec_hidden=decoder(dec_input,dec_hidden,enc_out)
        pred_id=tf,argmax(pred,axis=1)
        result+=targ_lang.idx2word[pred_id]+' '
        if targ_lang.idx2word[pred_id]=='<end>':
            return result,sentence
        dec_input=tf.expand_dims([pred_id], 1)
    return result,sentence


# In[ ]:


def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result,sentence=evaluate(sentene,encoder,decoder,inp_lang,targ_lang,max_length_inp,max_length_targ)
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))

