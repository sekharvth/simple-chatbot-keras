

# NOTE: The code below takes in the training data (Cornell Movie corpus) in it's entirety, and the memory requirement for 
# 	operations on the same will be enormous. I used a much smaller version of just 4500 context-target pairs, that had
#	total vocabulary size of around 6200 words, but I could only train for about 10 epochs on my CPU, wherein an accuracy
#	of only 60% was achieved. Needless to say, the outputs of queries posed to the trained chatbot were nowhere near remarkable.
#	This code, however, is the basic seq2seq implementation of the architecture of a chatbot, and with a few tweaks (such as 
#	using more LSTM layers, adding Dropout etc) and training till convergence, it shall give decent results


import numpy as np
import pickle
import operator

# load the data
context = np.load('context_indexes.npy')
final_target = np.load('target_indexes.npy')
with open('dictionary.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

# the indexes of the words start with 0. But when the sequences are padded later on, they too will be zeros.
# so, shift all the index values one position to the right, so that 0 is spared, and used only to pad the sequences
for i,j in word_to_index.items():
    word_to_index[i] = j+1

# reverse dictionary
index_to_word = {}
for k,v in word_to_index.items():
    index_to_word[v] = k

final_target_1 = final_target
context_1 = context

maxLen = 20

# shift the indexes of the context and target arrays too
for i in final_target_1:
    for pos,j in enumerate(i): i[pos] = j + 1
for i in context_1:
    for pos,j in enumerate(i): i[pos] = j + 1
	
# read in the 50 dimensional GloVe embeddings
def read_glove_vecs(file):
    with open(file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            word = line[0]
            words.add(word)
            word_to_vec_map[word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

words, word_to_vec_map = read_glove_vecs('.../data/glove.6B.50d.txt')

# since the indexes start from 1 and not 0, we add 1 to the no. of total words to get the vocabulary size (while initializing 
# and populating arrays later on, this will be required)
vocab_size = len(word_to_index) + 1

# initialize the embedding matrix that will be used (50 is the GloVe vector dimension)
embedding_matrix = np.zeros((vocab_size, 50))
for word,index in word_to_index_1.items():
    try:
        embedding_matrix[index, :] = word_to_vec_map[word.lower()]
    except: continue

# initialize and populate the outputs to the Keras model. The output is the same as the target, but shifted one time step to the left
# (teacher forcing)
outs = np.zeros((context_1.shape[0], maxLen, vocab_size))
for pos,i in enumerate(final_target_1):
    for pos1,j in enumerate(i):
        if pos1 > 0:
            outs[pos, pos1 - 1, j] = 1
    if pos%1000 == 0: print ('{} entries completed'.format(pos))

from keras.preprocessing import sequence

# pad the sequences so that they can be fed into the embedding layer
final_target_1 = sequence.pad_sequences(final_target_1, maxlen = 20, dtype = 'int32', padding = 'post', truncating = 'post')
context_1 = sequence.pad_sequences(context_1, maxlen = 20, dtype = 'int32', padding = 'post', truncating = 'post')


from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.models import Model

# load the pre-trained GloVe vectors into the embedding layer
embed_layer = Embedding(input_dim = vocab_size, output_dim = 50, trainable = True, )
embed_layer.build((None,))
embed_layer.set_weights([embedding_matrix])

# encoder and decoder gloabal LSTM variables with 300 units
LSTM_cell = LSTM(300, return_state = True)
LSTM_decoder = LSTM(300, return_state = True, return_sequences = True)
# final dense layer that uses TimeDistributed wrapper to generate 'vocab_size' softmax outputs for each time step in the decoder lstm
dense = TimeDistributed(Dense(vocab_size, activation = 'softmax'))

input_context = Input(shape = (maxLen, ), dtype = 'int32', name = 'input_context')
input_target = Input(shape = (maxLen, ), dtype = 'int32', name = 'input_target')

# pass the inputs into the embedding layer
input_ctx_embed = embed_layer(input_context)
input_tar_embed = embed_layer(input_target)

# pass the embeddings into the corresponding LSTM layers
encoder_lstm, context_h, context_c = LSTM_cell(input_ctx_embed)
# the decoder lstm uses the final states from the encoder lstm as the initial state
decoder_lstm, _, _ = LSTM_decoder(input_tar_embed, initial_state = [context_h, context_c],)

output = dense(vocab_size, activation = 'softmax')(decoder_lstm)

model = Model([input_context, input_target], output)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit([context_1, final_target_1], outs, epochs = 1000, batch_size = 128)

