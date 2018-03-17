import numpy as np
import re
from keras.preprocessing import sequence
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.models import Model

maxlen = 12
maxLen = 20

with open('reverse_dictionary.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

question = 'Hey! How are you doing?'
a = question.split()
for pos,i in enumerate(a):
    a[pos] = re.sub('[^a-zA-Z0-9 .,?!]', '', i)
    a[pos]= re.sub(' +', ' ', i)
    a[pos] = re.sub('([\w]+)([,;.?!#&\'\"-]+)([\w]+)?', r'\1 \2 \3', i)
    if len(i.split()) > maxlen:
            a[pos] = (' ').join(a[pos].split()[:maxlen])
            if '.' in a[pos]:
                ind = a[pos].index('.')
                a[pos] = a[pos][:ind+1]
            if '?' in a[pos]:
                ind = a[pos].index('?')
                a[pos] = a[pos][:ind+1]
            if '!' in a[pos]:
                ind = a[pos].index('!')
                a[pos] = a[pos][:ind+1]

question = ' '.join(a).split()

question = np.array([word_to_index[w] for w in question])

question = sequence.pad_sequences([question], maxlen = 20)

context_model = Model(input_context, [context_h, context_c])

target_h = Input(shape = (300, ))
target_c = Input(shape = (300, ))

target, h, c = LSTM_decoder(input_tar_embed, initial_state = [target_h, target_c])
output = dense(target)
target_model = Model([input_target, target_h, target_c], [output, h, c])



question_h, question_c = context_model.predict(question)

answer = np.zeros((1, maxLen))
answer[0, -1] = 1

i = 1
answer_1 = []
flag = 0
while flag != 1:
    prediction, prediction_h, prediction_c = target_model.predict([answer, question_h, question_c])
    token_arg = np.argmax(prediction[0, -1, :])
    answer_1.append(index_to_word[token_arg])
    if token_arg == 4 or token_arg == 0 or i > 20:
        flag = 1
    answer = np.zeros((1,maxLen))
    answer[0, -1] = token_arg
    question_h = prediction_h
    question_c = prediction_c
    i+=1
