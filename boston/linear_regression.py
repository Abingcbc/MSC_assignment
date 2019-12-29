from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import keras

import matplotlib.pyplot as plt
import numpy as np

housing_data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(housing_data.data, housing_data.target, test_size=0.2)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler().fit(y_train)

# 训练样本集、训练目标集 标准化
X_train = scalerX.transform(X_train)
y_train = scalery.transform(y_train)

# 测试样本集、测试目标集 标准化
X_test = scalerX.transform(X_test)

# 转为一维数组
y_train = y_train.reshape(-1,)

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
    m.add(Dense(1, kernel_regularizer=keras.regularizers.l1(0.1)))
    m.compile(optimizer='sgd', loss='mse')
    return m

history = LossHistory()
model = build_model()
model.fit(X_train, y_train, batch_size=64, epochs=50, callbacks=[history])
history.loss_plot('epoch')

y_pred = model.predict(X_test)
y_pred = scalery.inverse_transform(y_pred)

print(np.mean(np.abs(y_pred.reshape(1,-1)-y_test.reshape(1,-1))))
print('测试集R2分数：' + str(r2_score(y_pred=y_pred.reshape(1,-1)[0], y_true=y_test.reshape(1,-1)[0])))




