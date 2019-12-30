import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model, load_model
from keras_contrib.layers import CRF
import keras_contrib
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import gc

def read_data(path):
    data = []
    with open(path,'r',encoding='UTF-8') as f:
        for line in f.readlines():
            if '“' not in line:
                line = line.replace('”', '')
            elif '”' not in line:
                line = line.replace('“', '')
            elif '‘' not in line:
                line = line.replace('’', '')
            elif '’' not in line:
                line = line.replace('‘', '')
            data.append(line.split())
    return data

train = read_data('train.txt')
test = read_data('test.answer.txt')
print('rawdata:\n',train[0])

def add_label(data):
    x, y =[],[]
    for sentence in data:
        ch = []
        chlabel = [] # B-1 M-2 E-3 S-4
        for word in sentence:
            ch.extend(list(word))
            l = len(word)
            if l == 1:
                chlabel.append(4)
            elif l == 2:
                chlabel.extend([1, 3])
            elif l > 2:
                tmp = [2] * l
                tmp[0] = 1
                tmp[-1] = 3
                chlabel.extend(tmp)
        x.append(ch)
        y.append(chlabel)
    return x,y

xtrain, ytrain = add_label(train)
xtest, ytest = add_label(test)
x1test = xtest
print('labeled_data:\n', xtrain[0], '\n', ytrain[0])

SENLEN = 100
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(xtrain+xtest)
sequences = tokenizer.texts_to_sequences(xtrain+xtest)
word_index = tokenizer.word_index

xtrain = pad_sequences(sequences[:len(xtrain)], maxlen=SENLEN, padding='post', truncating='post', value=0)
ytrain = pad_sequences(ytrain, maxlen=SENLEN, padding='post', value=0)
xtrain = np.array(xtrain)
ytrain = keras.utils.to_categorical(np.array(ytrain), num_classes=5)

xtest = pad_sequences(sequences[len(xtrain):], maxlen=SENLEN, padding='post', truncating='post', value=0)
ytest = pad_sequences(ytest, maxlen=SENLEN, padding='post', value=0)
xtest = np.array(xtest)
ytest = keras.utils.to_categorical(np.array(ytest), num_classes=5)

print('num of characters: ', len(tokenizer.word_index))
print('xtrain_shape: ', xtrain.shape,' ytrain_shape: ', ytrain.shape)
print('xtest_shape: ', xtest.shape,' ytest_shape: ', ytest.shape)
print('encoded_data:\n', xtrain[0], '\n', ytrain[0][:5])

EMB_PATHS = 'glove.6B.300d.txt'

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def build_embedding_matrix(word_index, path):
    with open(path,'r',encoding='utf8') as f:
        embedding_index = dict(get_coefs(*line.strip().split(' ')) for line in f)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
        except:
            embedding_matrix[i] = embedding_index["unknown"]
    del embedding_index
    gc.collect()
    print('shape of embedding_matrix: ',embedding_matrix.shape)
    return embedding_matrix

# embedding_matrix = build_embedding_matrix(word_index, EMB_PATHS)

embdim = 300
maxlen = 100
hiddenDims = 100

# input = Input(shape=(maxlen,))
# emb = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=True)(input)
# lstm1 = Bidirectional(LSTM(hiddenDims, return_sequences=True))(emb)
# lstm2 = Bidirectional(LSTM(hiddenDims, return_sequences=True))(lstm1)
# lstm_out = Dropout(0.5)(lstm2)
# dense = TimeDistributed(Dense(5, activation='softmax'))(lstm_out)
# crf = CRF(5)(dense)
#
# model = Model(inputs=[input], outputs=[crf])
# model.summary()
#
# model.compile(optimizer='adam',
#               loss=keras_contrib.losses.crf_loss,
#               metrics=[keras_contrib.metrics.crf_accuracy])

custom_objects={'CRF': CRF,
                'crf_loss': keras_contrib.losses.crf_loss,
                'crf_accuracy': keras_contrib.metrics.crf_accuracy}
model = load_model('model.h5',custom_objects=custom_objects)
model.summary()

# model.fit(xtrain, ytrain, validation_data=(xtest, ytest), batch_size=512, epochs=10)
# model.save('model.h5')


def recall_m(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives)
    return recall

def precision_m(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives)
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))

ypred = model.predict(xtest)
print(f1_m(ytest,ypred),precision_m(ytest,ypred),recall_m(ytest,ypred))

# ypred2 = model.predict(xtrain)
# print(f1_m(ytrain,ypred2),precision_m(ytrain,ypred2),recall_m(ytrain,ypred2))

import pkuseg

seg = pkuseg.pkuseg()

res = []
for i in range(5):
    answer = ''
    sen = x1test[i]
    text = seg.cut(''.join(sen))  # 进行分词
    print(text)
    for j in range(min(100,len(sen))):
        label = ypred[i][j]
        label = np.argmax(label)
        answer += sen[j]
        if label == 3 or label == 4:
            answer += ' '
    res.append(answer)
print('\n'.join(res))

