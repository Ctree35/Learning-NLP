#!/usr/bin/env python
# coding: utf-8

# In[77]:


import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
tf.enable_eager_execution()
import numpy as np
import os
import time

# In[78]:


path=keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


# In[79]:


text=open(path,'r').read()
print(len(text))


# In[80]:


vocab=sorted(set(text))


# In[81]:


vocab_size=len(vocab)


# In[82]:


char2idx={u:i for i,u in enumerate(vocab)}
idx2char=np.array(vocab)
new_text=np.array([char2idx[i] for i in text])


# In[83]:


seq_length=100
chunks=tf.data.Dataset.from_tensor_slices(new_text).batch(seq_length+1, drop_remainder=True)


# In[84]:


def split_input(chunk):
    data=chunk[:-1]
    label=chunk[1:]
    return data,label


# In[85]:


dataset=chunks.map(split_input)


# In[86]:


batch_size=64
embedding_dim=256
units=1024
hidden_units=100
buffer_size=10000
dataset=dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)


# In[87]:


inputs=keras.Input(shape=(seq_length),batch_size=batch_size)
embed=keras.layers.Embedding(vocab_size,embedding_dim)(inputs)
gru=keras.layers.GRU(units, return_sequences=True, stateful=True)(embed)
relu=keras.layers.Dense(hidden_units, activation=tf.nn.relu)(gru)
out=keras.layers.Dense(vocab_size)(relu)


# In[88]:


model=keras.Model(inputs=inputs, outputs=out)


# In[89]:

def loss_function(real,pred):
    return tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(tf.to_int32(real),(-1,1)),logits=tf.reshape(pred,(-1,vocab_size)))


# In[90]:


model.compile(optimizer=keras.optimizers.Adam(0.01), loss=loss_function)


# In[ ]:


model.summary()


# In[ ]:


cdir='./checkpoints'
cpre=os.path.join(cdir, "ckpt")


# In[ ]:


epochs=5


# In[ ]:


model.fit(dataset, epochs=epochs)


# In[ ]:




