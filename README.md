# simple-chatbot-keras
Design and build a simple chatbot using data from the Cornell Movie Dialogues corpus, using Keras

Most of the ideas used in this model comes from the original seq2seq model made by the Keras team. It also serves as a brillant tutorial on the working of the architecture, and how it is developed: 
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

In short, the input sequence (the question asked to the chatbot) is passed into the encder LSTM, which outputs the final states of the encoder LSTM. These final states are passed into the decoder LSTM, along with the output sequence (the reply for the question, in the training data). The output of this decoder LSTM is the same as the actual reply, but shifted one time step to the left. That is, if the reply (aka, the input to the decoder lstm)  is 'I am fine', the output for first time step with input 'I' will be 'am', the input for the second time step will be 'am', with output 'fine', and so on.

In the inference mode, the 'BOS'(beginning of sentence) tag is the initial input to the decoder lstm, along with the final encoder states of the encoder lstm (obtained after passing new query into the encoder lstm). The output of this time step is used as input for the next time step, along with cell states of the current time step. This process repeats till 'EOS' tag (end of sentence) is generated.

But the model in the page above uses a character level model, which at first puzzled me, especially when most of the literature on the subject overwhelmingly adopted word level models. However, when I started with the word level model, I quickly found why the Keras team opted for the char level model.

When using word level models, the vocabulary (no. of unique words) of the enire data set (the Cornell Movie Dialogues corpus in this case) would be more than 50,000. And the number of examples for training amounted to ~300k (150000 pairs). 
When defining the outputs to the decoder lstm in the decoder model, the shape would be (num_examples, max_length_of_sentences, vocab_size). This would in effect, mean (150000, 20, 50000), which would raise memory errors.
When using the char level, instead of 50,000 for the vocab_size, it would reduce to something in the range of 70-80(26 for lowercase alphabets, 26 uppercase, 10 digits, unique symbols like '!', '?' etc), which would have better chances of going through without too many memory constraints.
The downside is that it will take an insane amount of epochs to converge, and can only be done on a powerful GPU, which is beyond my current capabilities.

The model shown here is the simplest of models, and for further improvement (definite requirement), more tweaking has to be done (increase the number of LSTM layers, introduce Dropout, play around with the optimizers etc)
