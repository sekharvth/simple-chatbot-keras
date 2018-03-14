import numpy as np
import pickle
import operator

f = open('movie_lines.txt', 'r')
lines = f.read().split('\n')
dic = {}
for line in lines:
    if len(line.split('+++$+++')) > 4:
        dic[int(line.split()[0][1:])] = line.split('+++$+++')[4:]
		

lst = sorted(dic.items(), key = operator.itemgetter(0))
batches = {}
count = 1
batch = []
for i in range(1, len(lst) + 1):
    if i < len(lst):
        if lst[i][0] == lst[i-1][0] + 1:
            if lst[i-1][1][0].lstrip() not in batch : 
                batch.append(lst[i-1][1][0].lstrip()) 
            batch.append(lst[i][1][0].lstrip()) 
        else:
            batches[count] = batch
            batch = []
        count+=1
    else:
        pass
    
context_and_target = []
for ls in batches.values():
    if len(ls)%2!=0: ls = ls[:-1]
    for i in range(0, len(ls), 2):
        context_and_target.append((ls[i], ls[i+1]))
		
context, target = zip(*context_and_target)

target = list(target)

import re
maxlen = 12
for pos,i in enumerate(target):
    target[pos] = re.sub('[^a-zA-Z0-9 .,?!]', '', i)
    target[pos] = re.sub(' +', ' ', i)
    target[pos] = re.sub('([\w]+)([,;.?!#&-\'\"-]+)([\w]+)?', r'\1 \2 \3', i)
    if len(i.split()) > maxlen:
        target[pos] = (' ').join(target[pos].split()[:maxlen])
        if '.' in target[pos]:
            ind = target[pos].index('.')
            target[pos] = target[pos][:ind+1]
        if '?' in target[pos]:
            ind = target[pos].index('?')
            target[pos] = target[pos][:ind+1]
        if '!' in target[pos]:
            ind = target[pos].index('!')
            target[pos] = target[pos][:ind+1]

context = list(context)
for pos,i in enumerate(context):
    context[pos] = re.sub('[^a-zA-Z0-9 .,?!]', '', i)
    context[pos] = re.sub(' +', ' ', i)
    context[pos] = re.sub('([\w]+)([,;.?!#&\'\"-]+)([\w]+)?', r'\1 \2 \3', i)
    if len(i.split()) > maxlen:
            context[pos] = (' ').join(context[pos].split()[:maxlen])
            if '.' in context[pos]:
                ind = context[pos].index('.')
                context[pos] = context[pos][:ind+1]
            if '?' in context[pos]:
                ind = context[pos].index('?')
                context[pos] = context[pos][:ind+1]
            if '!' in context[pos]:
                ind = context[pos].index('!')
                context[pos] = context[pos][:ind+1]
				
final_target = ['BOS '+i+' EOS' for i in target]

final_target = list(pd.Series(final_target).map(lambda x: re.sub(' +', ' ', x)))
context = list(pd.Series(context).map(lambda x: re.sub(' +', ' ', x)))

counts = {}
for words in final_target+context:
    for word in words.split():
        counts[word] = counts.get(word,0) + 1
		
word_to_index = {}
for pos,i in enumerate(counts.keys()):
    word_to_index[i] = pos
	

final_target = np.array([[word_to_index[w] for w in i.split()] for i in final_target])
context = np.array([[word_to_index[w] for w in i.split()] for i in context])

np.save('context_indexes', context)

np.save('target_indexes', final_target)

with open('dictionary.pkl', 'wb') as f:
    pickle.dump(word_to_index, f, pickle.HIGHEST_PROTOCOL)
	
context = np.load('context_indexes.npy')
final_target = np.load('target_indexes.npy')
with open('dictionary.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

len(word_to_index)

for i,j in word_to_index.items():
    word_to_index[i] = j+1
word_to_index

index_to_word = {}
for k,v in word_to_index.items():
    index_to_word[v] = k

final_target_1 = final_target[:4500]
context_1 = context[:4500]

final_target_1

for i in final_target_1:
    for pos,j in enumerate(i): i[pos] = j + 1
for i in context_1:
    for pos,j in enumerate(i): i[pos] = j + 1

word_to_index_1 = {}
lst = []
for i in final_target_1:
    for j in i:
        lst.append(index_to_word[j])
for i in context_1:
    for j in i:
        lst.append(index_to_word[j])
		
for pos, j in enumerate(list(set(lst))):
    word_to_index_1[j] = pos

for i in final_target_1:
    for pos,j in enumerate(i):
        i[pos] = word_to_index_1[index_to_word[j]]
for i in context_1:
    for pos,j in enumerate(i):
        i[pos] = word_to_index_1[index_to_word[j]]

import numpy as np
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

words, word_to_vec_map = read_glove_vecs('/home/jovyan/work/Week 2/Word Vector Representation/data/glove.6B.50d.txt')

vocab_size = len(word_to_index_1)
embedding_matrix = np.zeros((vocab_size, 50))
for word,index in word_to_index_1.items():
    try:
        embedding_matrix[index, :] = word_to_vec_map[word.lower()]
    except: continue

from keras.preprocessing import sequence
#from keras.preprocessing.text import Tokenizer
final_target_1 = sequence.pad_sequences(final_target_1, maxlen = 20, dtype = 'int32', padding = 'post', truncating = 'post')
context_1 = sequence.pad_sequences(context_1, maxlen = 20, dtype = 'int32', padding = 'post', truncating = 'post')

outs = np.zeros((4500, 20, vocab_size))
for pos,i in enumerate(final_target_1):
    for pos1,j in enumerate(i):
        if pos1 > 0:
            outs[pos, pos1 - 1, j] = 1
    if pos%1000 == 0: print ('{} entries completed'.format(pos))
	
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM
from keras.models import Model
import keras.backend as K
import tensorflow as tf

embed_layer = Embedding(input_dim = vocab_size, output_dim = 50, trainable = True, )
embed_layer.build((None,))
embed_layer.set_weights([embedding_matrix])

outputs = []

LSTM_cell = LSTM(300, return_state = True)
LSTM_decoder = LSTM(300, return_state = True, return_sequences = True)

input_context = Input(shape = (20, ), dtype = 'int32', name = 'input_context')
input_target = Input(shape = (20, ), dtype = 'int32', name = 'output_context')

input_ctx_embed = embed_layer(input_context)
input_tar_embed = embed_layer(input_target)

encoder_lstm, context_h, context_c = LSTM_cell(input_ctx_embed)
decoder_lstm, _, _ = LSTM_decoder(input_tar_embed, initial_state = [context_h, context_c],)

output = Dense(vocab_size, activation = 'softmax')(decoder_lstm)
outputs.append(output)

model = Model([input_context, input_target], output)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit([context_1, final_target_1], outs, epochs = 100, batch_size = 128)
