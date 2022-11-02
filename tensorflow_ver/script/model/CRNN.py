from PIL.Image import merge
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D,ZeroPadding2D,Add,GlobalAveragePooling1D,GlobalAveragePooling2D, ReLU,Reshape,Lambda,BatchNormalization,Multiply,AveragePooling2D,AveragePooling1D
from keras.layers.merge import add, concatenate, Concatenate
from keras.layers.recurrent import LSTM
from keras.layers import CuDNNLSTM
from keras import regularizers
import os


def crnn(input):
     # Convolution layer
    block1 = Conv2D(96, (5,1), padding='same', kernel_initializer='he_normal', name='conv1')(input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)
    block1 = AveragePooling2D(pool_size=(5,1), name = 'f_max1')(block1)

    block2 = Conv2D(96, (3,1), padding='same', kernel_initializer='he_normal', name='conv2')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)
    block2 = AveragePooling2D(pool_size=(2,1), name = 'f_max2')(block2)

    block3 = Conv2D(96, (3,1), padding='same', kernel_initializer='he_normal', name='conv3')(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)
    block3 = AveragePooling2D(pool_size=(2,1), name = 'f_max3')(block3)

    block4 = Conv2D(96, (3,1), padding='same', kernel_initializer='he_normal', name='conv4')(block3)
    block4 = BatchNormalization()(block4)
    block4 = Activation('relu')(block4)
    block4 = AveragePooling2D(pool_size=(2,1), name = 'f_max4')(block4)

    # CNN to RNN
    inner_w = Reshape((int(block4.shape[2]), int(block4.shape[1]) * int(block4.shape[3])), name='reshape_w')(block4)  # (None, 96,40)    

    # RNN layer
    lstm_w  = CuDNNLSTM(96, return_sequences=True, name='lstm_w')(inner_w)
    lstm_back_w = CuDNNLSTM(96, return_sequences=True,go_backwards=True, name='lstm_w_back')(inner_w)
    lstm_add_w = concatenate([lstm_w,lstm_back_w])
    inner = BatchNormalization()(lstm_add_w)
    inner = GlobalAveragePooling1D()(inner)
    inner = BatchNormalization()(inner)
    output = Dense(100,activation='relu')(inner)

    return output