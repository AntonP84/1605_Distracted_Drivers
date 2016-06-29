from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, BaseLogger, History, Callback

import numpy as np

class LossHistory(Callback):
    def __init__(self, patience):
        super(Callback, self).__init__()
        self.patience = patience

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.loss = np.inf
        self.val_loss = np.inf

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
    
    def on_train_end(self, logs={}):
        self.val_loss = min(self.val_losses[-(self.patience+2):])
        self.loss = self.losses[-(self.patience+2):][self.val_losses[-(self.patience+2):].index(self.val_loss)]
        
def create_model(img_rows, img_cols, color_type=3, nb_filters = 16, pct_dropout = 0.5):
        
    model = Sequential()
    model.add(Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(pct_dropout))

#     model.add(Convolution2D(nb_filters*2, 3, 3, activation='relu', border_mode='same'))
#     model.add(Convolution2D(nb_filters*2, 3, 3, activation='relu', border_mode='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(pct_dropout))

#     model.add(Convolution2D(nb_filters*4, 3, 3, activation='relu', border_mode='same'))
#     model.add(Convolution2D(nb_filters*4, 3, 3, activation='relu', border_mode='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(pct_dropout))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(Adam(), loss='categorical_crossentropy')
    return model