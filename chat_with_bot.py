# This is the inference model. After training, a different model architecture is used to make predictions on new input data, 
# using the trained layers such as LSTM and TimeDistributed Dense which were defined globally during training.
# The code from chat_bot.py (particualrly the code pertaining to training the model) has to be pasted into this part. I skipped this part
# to make the entire code simpler to understand.

# import required packages
import numpy as np
import re
from keras.preprocessing import sequence
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.models import Model

# for initial filtering
maxlen = 12
maxLen = 20

# import the dictionary
with open('dictionary.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

# import the reverse dictionary
with open('reverse_dictionary.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

# the question asked to the chatbot
question = 'Hey! How are you doing?'

# preprocessing to make the data into the format required by the model, same as during training
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

# make the question into an array of the corresponding indexes
question = np.array([word_to_index[w] for w in question])

# pad sequences
question = sequence.pad_sequences([question], maxlen = 20)

# Keras model used to train, so that we define the variables (tensors) that ultimately go into the infernce model
input_context = Input(shape = (maxLen, ), dtype = 'int32', name = 'input_context')
input_target = Input(shape = (maxLen, ), dtype = 'int32', name = 'output_context')

input_ctx_embed = embed_layer(input_context)
input_tar_embed = embed_layer(input_target)

encoder_lstm, context_h, context_c = LSTM_encoder(input_ctx_embed)
decoder_lstm, h, _ = LSTM_decoder(input_tar_embed, initial_state = [context_h, context_c],)

output = dense(decoder_lstm)

# Define the model for the input (question). Returns the final state vectors of the encoder LSTM
context_model = Model(input_context, [context_h, context_c])

# define the inputs for the decoder LSTM
target_h = Input(shape = (300, ))
target_c = Input(shape = (300, ))

# the decoder LSTM. Takes in the embedding of the initial word passed as input into the decoder model (the 'BOS' tag), 
# along with the final states of the encoder model, to output the corresponding sequences for 'BOS', and the new LSTM states.  
target, h, c = LSTM_decoder(input_tar_embed, initial_state = [target_h, target_c])
output = dense(target)
target_model = Model([input_target, target_h, target_c], [output, h, c])

# pass in the question to the encoder LSTM, to get the final encoder states of the encoder LSTM
question_h, question_c = context_model.predict(question)

# initialize the answer that will be generated for the 'BOS' input. Since we have used pre-padding for padding sequences,
# the last token in the 'answer' variable is initialised with the index for 'BOS'.
answer = np.zeros((1, maxLen))
answer[0, -1] = word_to_index['BOS']

# i keeps track of the length of the generated answer. This won't allow the model to genrate sequences with more than 20 words.
i = 1

# make a new list to store the words generated at each time step
answer_1 = []

# flag to stop the model when 'EOS' tag is generated or when 20 time steps have passed.
flag = 0

# run the inference model
while flag != 1:
    # make predictions for the given input token and encoder states
    prediction, prediction_h, prediction_c = target_model.predict([answer, question_h, question_c])
    
    # from the generated predictions of shape (num_examples, maxLen, vocab_size), find the token with max probability
    token_arg = np.argmax(prediction[0, -1, :])
    
    # append the corresponding word of the index to the answer_1 list
    answer_1.append(index_to_word[token_arg])
    
    # set flag to 1 if 'EOS' token is generated or 20 time steps have passed
    if token_arg == word_to_index['EOS'] or i > 20:
        flag = 1
    # re-initialise the answer variable, and set the last token to the output of the current time step. This is then passed
    # as input to the next time step, along with the LSTM states of the current time step
    answer = np.zeros((1,maxLen))
    answer[0, -1] = token_arg
    question_h = prediction_h
    question_c = prediction_c
    
    # increment the count of the loop
    i+=1
    
 # print the answer generated for the given question
print (' '.join(answer_1))
