import keras
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
import lda2vec
import string
import re

# ^ = [\w|\s|'|<|>|@]+ = $

printable = set(string.printable)
punctuation = set(string.punctuation)

def wordIsPrintable(word):
    printableWord = True
    for c in word:
        if c not in printable:
            printableWord = False
            break
    return printableWord

if __name__ == '__main__':

    train_vocab = set()
    with open('wikitext-2/wiki.train.tokens', 'r') as train:
        for line in train:
            words = line.split()
            for word in words:
                if word not in train_vocab:
                    train_vocab.add(word)

    currentTopic = ''
    topicsAndDocuments = []
    with open('wikitext-2/wiki.train.tokens', 'r') as train:
        for line in train:
            if line is not '\n':
                if line.find('=') == -1:
                    newLine = line
                    # newLine = newLine.replace('\n', '')
                    # newLine = newLine.replace('=', '')
                    # newLine = newLine.replace('<unk>', '')
                    # newLine = newLine.replace('@-@', '')
                    # newLine = newLine.replace('-', '')
                    # newLine = newLine.translate(str.maketrans('', '', string.punctuation))
                    # newLine = ''.join(list(filter(lambda x: x in printable, newLine)))
                    # newLine = newLine.strip()
                    if newLine != ' ' and newLine != '':
                        topicsAndDocuments.append((currentTopic, newLine))
                else:
                    newLine = line
                    newLine = newLine.replace('\n', '')
                    newLine = newLine.replace('=', '')
                    newLine = newLine.replace('<unk>', '')
                    newLine = newLine.replace('@-@', '')
                    newLine = newLine.replace('-', '')
                    newLine = newLine.translate(str.maketrans('', '', string.punctuation))
                    newLine = ''.join(list(filter(lambda x: x in printable, newLine)))
                    newLine = newLine.strip()
                    currentTopic = newLine

    tokens = ''
    for topic, document in topicsAndDocuments:
        tokens += document

    tokens = tokens.split(' ')

    tokens = list(filter(lambda x: x != '<unk>', tokens))
    tokens = list(filter(lambda x: x != '@-@', tokens))
    tokens = list(filter(lambda x: x not in punctuation, tokens))
    tokens = list(filter(lambda x: wordIsPrintable(x), tokens))

    length = 50 + 1
    sequences = []
    for i in range(length, len(tokens)):
        seq = tokens[i-length:i]
        line = ''
        for word in seq:
            line += word + ' '
        line = line.strip()
        sequences.append(line)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)

    sequences = list(filter(lambda x: len(x) == length, sequences))

    print('Total Sequences: %d' % len(sequences))

    sequencesTemp = []
    for s in sequences:
        sequencesTemp.append(np.array(s))

    sequences = np.array(sequencesTemp)

    vocab_size = len(tokenizer.word_index) + 1

    X, y = sequences[:,:-1], sequences[:,-1]
    y = keras.utils.to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 50, input_length=seq_length))
    model.add(keras.layers.LSTM(100, return_sequences=True))
    model.add(keras.layers.LSTM(100))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(vocab_size, activation='softmax'))
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(X, y, batch_size=128, epochs=100)

    # save the model to file
    model.save('model.h5')
    # save the tokenizer
    dump(tokenizer, open('tokenizer.pkl', 'wb'))
