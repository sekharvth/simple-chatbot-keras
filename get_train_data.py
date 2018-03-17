import numpy as np
import pickle
import operator

# read in the Cornell Movie Dialogues data
f = open('movie_lines.txt', 'r')
lines = f.read().split('\n')
dic = {}
for line in lines:
    if len(line.split('+++$+++')) > 4:
        dic[int(line.split()[0][1:])] = line.split('+++$+++')[4:]
		
# sort the dialogues into the proper sequence based on the line number 'L...' in the data
lst = sorted(dic.items(), key = operator.itemgetter(0))

# make the queries and replies into different batches based on the films in the data set
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
 
# make the data into context and target pairs
context_and_target = []
for ls in batches.values():
    if len(ls)%2!=0: ls = ls[:-1]
    for i in range(0, len(ls), 2):
        context_and_target.append((ls[i], ls[i+1]))
		
context, target = zip(*context_and_target)

target = list(target)

# do some basic preprocessing, filter out dialogues with more than 12 words, and in the 12 or lesser words, take only the characters
# till one of '!' or '.' or '?' comes
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

# add Beginning of Sentence (BOS) and End of Sentence (EOS) tags to the 'target' data
final_target = ['BOS '+i+' EOS' for i in target]

# remove any extra spaces
final_target = list(pd.Series(final_target).map(lambda x: re.sub(' +', ' ', x)))
context = list(pd.Series(context).map(lambda x: re.sub(' +', ' ', x)))

# get all the unique words in the data set with their counts
counts = {}
for words in final_target+context:
    for word in words.split():
        counts[word] = counts.get(word,0) + 1
	
# make the dictionary mapping words to indexes
word_to_index = {}
for pos,i in enumerate(counts.keys()):
    word_to_index[i] = pos

# reverse dictionary mapping indexes to words
index_to_word = {}
for k,v in word_to_index.items():
    index_to_word[v] = k	

# apply the dictionary to the context and target data
final_target = np.array([[word_to_index[w] for w in i.split()] for i in final_target])
context = np.array([[word_to_index[w] for w in i.split()] for i in context])

# save files
np.save('context_indexes', context)

np.save('target_indexes', final_target)

with open('dictionary.pkl', 'wb') as f:
    pickle.dump(word_to_index, f, pickle.HIGHEST_PROTOCOL)

with open('reverse_dictionary.pkl', 'wb') as f:
    pickle.dump(index_to_word, f, pickle.HIGHEST_PROTOCOL)
	
