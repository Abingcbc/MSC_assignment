from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import keras

import matplotlib.pyplot as plt
import numpy as np

iris_data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

scalerX = StandardScaler().fit(X_train)
encoder = OneHotEncoder().fit(y_train)

X_train = scalerX.transform(X_train)
y_train = encoder.transform(y_train).toarray()

X_test = scalerX.transform(X_test)
y_test = encoder.transform(y_test).toarray()


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def build_model():
    m = Sequential()
    m.add(Dense(3, kernel_regularizer=keras.regularizers.l1(0.1), activation='softmax'))
    m.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

history = LossHistory()
model = build_model()
model.fit(X_train, y_train, batch_size=32, epochs=200, callbacks=[history])
history.loss_plot('epoch')

y_pred = model.predict(X_test)
y_pred = [np.argmax(y) for y in y_pred]
y_test = [np.argmax(y) for y in y_test]
print(y_pred)
print(y_test)
print(classification_report(y_test, y_pred))


