import scipy.io as sio
import numpy as np
# 导入顺序模型
from keras.models import Sequential
# 导入全连接层Dense， 激活层Activation 以及 Dropout层
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import random


def loadData(fileName):
    data = sio.loadmat(fileName)
    X = data['X']
    y = data['y']
    # print("X.shape:",X.shape)
    # print("y.shape:",y.shape)
    return X, y


def train(trainSetFile):
    X_train, Y_train = loadData('train_set.mat')
    Y_train = Y_train.T.reshape(-1)  # 将Y_train转为序为1，即成为向量
    Y_train[Y_train == 10] = 0
    Y_train = convert_to_one_hot(Y_train, 10)  # 将Y_train转为one hot数据形式
    #print(Y_train[4999:])
    #print(X_train.shape, Y_train.shape)
    np.random.seed(1337)

    # 建立顺序型模型
    model = Sequential()

    '''
    模型需要知道输入数据的shape，
    因此，Sequential的第一层需要接受一个关于输入数据shape的参数，
    后面的各个层则可以自动推导出中间数据的shape，
    因此不需要为每个层都指定这个参数
    '''
    # 输入层有400个神经元
    # 第一个隐层有256个神经元，激活函数为ReLu，Dropout比例为0.2
    model.add(Dense(256, input_shape=(400,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # 第二个隐层有128个神经元，激活函数为ReLu，Dropout比例为0.2
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # 输出层有10个神经元，激活函数为SoftMax，得到分类结果
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # 输出模型的整体信息
    # 总共参数数量为784*512+512 + 512*512+512 + 512*10+10 = 669706
    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    history = model.fit(X_train, Y_train, batch_size=256, epochs=20)

    plot_model(model, to_file='model.png')
    SVG(model_to_dot(model).create(prog='dot', format='svg')) #在ipython中展示模型图片
    model.save('my_model.h5')  #保存keras模型与参数
    print("对训练集已训练好，训练模型及参数已保存！")
    return model


def convert_to_one_hot(Y, C):  # 将Y转为one hot形式，参数C代表分类数
    Y = np.eye(C)[Y.reshape(-1)]  # 这里-1是让序成为1的一种简便方法
    return Y

def myPredict(model,X,needNorm=True):
    # Find min and max grays values in the image
    maxValue = np.max(X)
    minValue = np.min(X)
    # Compute the value range of actual grays
    delta = maxValue - minValue
    if(needNorm):
        # Normalize grays between 0 and 1
        X = (X - minValue) / delta #在实测中，对输入图像标准化好还是不标准化好，有点吃不准；
    # print(X)
    # print(X.shape)
    return model.predict(X)

def selOneData(trainSetFile):
    # 随机抽出一个训练样本
    X, y = loadData(trainSetFile)
    m = X.shape[0]
    sel = random.randint(0, m - 1)  # sel为随机整数，其值范围为: 0<=sel<=m-1
    return X[sel], y[sel]