#!/usr/bin/env python
# coding: utf-8

# In[2]:



import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.neural_network._stochastic_optimizers import *
import random
import sys
sys.path.append('/csec/project/')
from wrap_client import *


    


# In[3]:



## generate a test model
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import metrics

x_train=pd.read_csv('training_set.csv')
y_train=x_train['Class']
x_train= x_train[x_train.columns[~x_train.columns.isin(['Class'])]]

x_test=pd.read_csv('test_set.csv')
y_test = x_test['Class']
x_test= x_test[x_test.columns[~x_test.columns.isin(['Class'])]]


# In[4]:


model=LogisticRegression(solver='lbfgs')
model.fit(x_train,y_train)


y_pred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy test original: {:.2f}'.format(accuracy))


# In[5]:


## calling the first api to save the model
modelkey = save(model)
print("model_key",modelkey)


# In[5]:



model_new = get(modelkey)


y_pred = model_new.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy test from saved model: {:.2f}'.format(accuracy))


# In[34]:


print(model_new)


# In[6]:


test_idx = random.choice(np.where(y_pred==0)[0])
print(test_idx)
test_data = np.array(x_test.values[test_idx])
test_res = y_pred[test_idx]
print(test_data)
print("original test result ",test_res)

## corruputed model
model.coef_=np.array([[57483.3923838202,475847.350293178,98484.3659165597,110.1578489449,8940.006635965,78432.3229955366,8394.3830462388,0.6772012382,0.3201764176,-0.3041301055,-0.2341952523,0.573061855,-0.2139525214,0.7198113618,0.2532606831,-0.1168181896,-0.3184781116,-0.2908866759,-0.4234341481,-0.5401860055]])



corrupted_res = model.predict(test_data.reshape(1, -1))
print('corrupted model result', corrupted_res)


# In[7]:


valid_res = verify(modelkey,test_data,corrupted_res)
print('is model corrupted', valid_res )


# In[12]:


from sklearn.neural_network import MLPClassifier
## Next let's try using a neural network.

nn_model = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='adam', alpha=0.01,learning_rate_init=0.001, max_iter=800)
nn_model.fit(x_test,y_test)
y_pred=nn_model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# In[13]:


model_nn_key = save(nn_model)
print(model_nn_key)


# In[10]:


model_nn_new = get(model_nn_key)
y_pred = model_nn_new.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy test: {:.2f}'.format(accuracy))


# In[11]:


test_idx = random.choice(np.where(y_pred==0)[0])
print(test_idx)
test_data = np.array(x_test.values[test_idx])
test_res = y_pred[test_idx]
print(test_data)
print("original test result ",test_res)


# In[78]:


#model_nn_new.coefs_[0][2]=879.09

corrupted_res = model_nn_new.predict(test_data.reshape(1, -1))
print('not corrupted model result', corrupted_res)


# In[89]:


valid_res = verify(model_nn_key,test_data,corrupted_res)
print('is model corrupted', valid_res )


# In[32]:





# In[29]:




# In[56]:


lb =getLabelBinarizer()
lb.__dict__


# In[1]:


## try an RNN

from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time


# In[27]:


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))

# Take a look at the first 250 characters in text
print(text[:250])

# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

    
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))   
    
# Batch size 
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences, 
# so it doesn't attempt to shuffle the entire sequence in memory. Instead, 
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension 
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

if tf.test.is_gpu_available():
  rnn = tf.keras.layers.CuDNNGRU
else:
  import functools
  rnn = functools.partial(
    tf.keras.layers.GRU, recurrent_activation='sigmoid')

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                              batch_input_shape=[batch_size, None]),
    rnn(rnn_units,
        return_sequences=True, 
        recurrent_initializer='glorot_uniform',
        stateful=True),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab), 
  embedding_dim=embedding_dim, 
  rnn_units=rnn_units, 
  batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1): 
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)") 
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=1


# In[28]:


history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])


# In[29]:


tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
model.summary()
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing) 
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
      
      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"ROMEO: "))
model = build_model(
  vocab_size = len(vocab), 
  embedding_dim=embedding_dim, 
  rnn_units=rnn_units, 
  batch_size=BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()

# Training step
EPOCHS = 1

for epoch in range(EPOCHS):
    start = time.time()
    
    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()
    
    for (batch_n, (inp, target)) in enumerate(dataset):
          with tf.GradientTape() as tape:
              # feeding the hidden state back into the model
              # This is the interesting step
              predictions = model(inp)
              loss = tf.losses.sparse_softmax_cross_entropy(target, predictions)
              
          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(grads, model.trainable_variables))

          if batch_n % 100 == 0:
              template = 'Epoch {} Batch {} Loss {:.4f}'
              print(template.format(epoch+1, batch_n, loss))

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
      model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))


# In[33]:


model_rnn_key = save(model)


# In[ ]:




