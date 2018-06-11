import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import os
import importlib

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend


#AlexNet with batch normalization in Keras
#input image is 224x224


def main():

    ## tensorflow


    ## keras tensorflow
    model = Sequential()

    model.add(Convolution2D(96, (11, 11), subsample=(4, 4), input_shape=(227, 227, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Convolution2D(256, (5, 5), subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Convolution2D(384, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Convolution2D(384, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    # model.add(Dense(12*12*256, 4096, init='normal'))
    # model.add(BatchNormalization(4096))
    # model.add(Activation('relu'))
    model.add(keras.layers.SimpleRNN(1, input_shape=(1, 9216, 1)))
    model.add(Dense(4096, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 1000, init='normal'))
    model.add(BatchNormalization(1000))
    model.add(Activation('softmax'))

    ## thenao
    # model = Sequential()
    #
    # model.add(Convolution2D(96, (11, 11), padding='full', input_shape=(227, 227, 3)))
    # # model.add(BatchNormalization(input_shape=(64, 226, 226)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(4, 4)))
    #
    # model.add(Convolution2D(256, (7, 7), padding='full', input_shape=(226, 226, 64)))
    # # model.add(BatchNormalization(input_shape=(128, 115, 115)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    #
    # model.add(Convolution2D(384, (3, 3), padding='full', input_shape=(115, 115, 128)))
    # # model.add(BatchNormalization(input_shape=(128, 112, 112)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    #
    # model.add(Convolution2D(384, (3, 3), padding='full', input_shape=(115, 115, 128)))
    # # model.add(BatchNormalization(input_shape=(128, 112, 112)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    #
    # model.add(Convolution2D(256, (3, 3), padding='full', input_shape=(112, 112, 64)))
    # # model.add(BatchNormalization(input_shape=(19, 108, 108)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(3, 3)))
    #
    # model.add(Flatten())
    # # model.add(Dense(12*12*256, 4096, init='normal'))
    # # model.add(BatchNormalization(4096))
    # # model.add(Activation('relu'))
    # model.add(keras.layers.SimpleRNN(1))
    # model.add(Dense(4096, 4096, init='normal'))
    # model.add(BatchNormalization(4096))
    # model.add(Activation('relu'))
    # model.add(Dense(4096, 1000, init='normal'))
    # model.add(BatchNormalization(1000))
    # model.add(Activation('softmax'))


if __name__ == "__main__":
    # set_keras_backend("theano")
    main()
