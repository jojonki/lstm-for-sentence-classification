# lstm-for-sentence-classification

A sentence classification (0/1) model using LSTM. Given one sentence (word list) which is embedded, then fed to LSTM. Finally it outputs 0/1 via Dense layer.

Related work in CNN version is [here](https://github.com/jojonki/cnn-for-sentence-classification). 


This is a model's summary.
```
embd_size = 128
hidden_size = 64
sentence_input = Input(shape=(None, ))
embd_sentence = Embedding(input_dim=vocab_size, output_dim=embd_size, input_length=sentence_maxlen)(sentence_input)
embd_sentence = Dropout(0.2)(embd_sentence)
out_rnn = LSTM(hidden_size, dropout=0.2, recurrent_dropout=0.2)(embd_sentence)
output = Dense(1, activation='sigmoid')(out_rnn)
model = Model(sentence_input, output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

------
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, None)              0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 42, 128)           296320    
_________________________________________________________________
dropout_1 (Dropout)          (None, 42, 128)           0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 64)                49408     
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 65        
=================================================================
```

## dataset
You need to download `training.txt` from [UMICH SI650 - Sentiment Classification](https://www.kaggle.com/c/si650winter11/data).
