# config coding=utf-8
import cv2
import numpy as np
import datetime
import os
import scipy.io as sio
from keras.models import Model, load_model
import my_utils

drawing = False  # 鼠标左键按下时，该值为True，标记正在绘画
mode = False  # True 画矩形，False 自由笔画
ix, iy = -1, -1  # 鼠标左键按下时的坐标
c_width = 200  # 画布宽
c_height = 200  # 画布高
line_width = 25  # 自由笔画线宽
isGrayMode = False  # 当前图像显示模式是否为灰度模式
tip = "\nm:切换画笔  c:清除画笔 g:将图像转为灰度 s:保存图像 a:增加样本 t:训练样本 i:识别 e:样本测试  ESC:退出"
trainSetFile = "train_set(样本数5030).mat"
if os.access("my_model.h5", os.F_OK):
    model = load_model('my_model.h5')  #载入keras模型与参数
    print("发现kreas模型与参数文件并调入内存!")
else:
    print("没有发现kreas模型与参数文件，请先对样本进行训练！")


# all_theta = np.zeros((10, 401))

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        # 鼠标左键按下事件
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        # 鼠标移动事件
        if drawing == True:
            if mode == True:
                # 最后一个参数是线条粗细，-1为自动填充图形
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                # cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.line(img, (ix, iy), (x, y), (255, 255, 255), line_width)
                ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        # 鼠标左键松开事件
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            # cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.line(img, (ix, iy), (x, y), (255, 255, 255), line_width)


img = np.zeros((c_width, c_height, 3), np.uint8)
cv2.namedWindow('image')  # 接着创建一个窗口
cv2.setMouseCallback('image', draw_circle)  # 设置鼠标事件的回调函数

print(tip)
while (1):
    cv2.imshow('image', img)  # 在窗口中显示图像
    k = cv2.waitKey(1) & 0xFF  # 按键等待时间为1ms
    if k == ord('m'):  # ord以一个字符（长度为1的字符串）作为参数，返回对应的ASCII数值，或者Unicode数值
        mode = not mode
        if mode:
            print("当前画笔：矩形")
        else:
            print("当前画笔：自由")
    if k == ord('g'):  # 按g转为将图像模式转为灰度
        if not isGrayMode:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            isGrayMode = True
            # print(img.shape)
            print("图像模式已为灰度模式！")
            print(tip)
    elif k == ord('c'):  # 清除画面
        img = img * 0
    elif k == 115:  # 按s键保存当前图片
        nowTime = datetime.datetime.now().strftime('%y%m%d%H%M%S')
        # print(nowTime)
        cv2.imwrite(nowTime + ".jpg", img)
        print("图像已保存！")
        print(tip)
    elif k == ord('a'):
        print("先手画好数字图像后，再按a后键入对应数字,按esc放弃：")
        print("请输入相应数字：")
        k = -1
        while (k < 0):  # 循环等待键盘输入
            k = cv2.waitKey(1)
        # print(k)
        k = k & 0xFF
        if k == 27:
            print("放弃操作！")
        elif ord('0') <= k <= ord('9'):
            k = k - 48  # 将k转为对应数值
            print(k)
            if k == 0:
                k = 10  # 样本0对应标签值为10
            if isGrayMode:
                img_c = img
                # print(img_c.shape)
            else:
                img_c = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将img_c转成灰度图
            img_c = cv2.resize(img_c, (20, 20))  # 设定img_c为识别标准尺寸
            img_c = img_c / 255

            if os.access(trainSetFile, os.F_OK):
                data = sio.loadmat(trainSetFile)
                X = data['X']
                y = data['y']
                print("发现训练样本集文件！")
                X = np.row_stack((X, img_c.reshape(1, 400, order='F')))
                y = np.row_stack((y, np.array([[k]])))
                print("X", X)
                print("y", y)
                sio.savemat(trainSetFile, {"X": X, "y": y})
                print("新样本已保存入训练集文件！")
                print("现有样本数:", X.shape[0])
            else:
                print("没有发现训练样本文件！")
                sio.savemat(trainSetFile, {
                            "X": img_c.reshape(1, 400, order='F'), "y": k})
                print("已经创建训练样本集文件，并将当前样本存入其中！")
        else:
            # print(k)
            print("因为键入非数字符，操作已放弃！")
        print(tip)
    elif k == ord('t'):
        model=my_utils.train(trainSetFile)  #用keras神经模型训练样本集
        print(tip)
    elif k == ord('i'):
        if isGrayMode:
            img_c = img
            # print(img_c.shape)
        else:
            img_c = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将img_c转成灰度图
        img_c = cv2.resize(img_c, (20, 20))  # 设定img_c为识别标准尺寸
        # img = img/255
        # print(img_c.shape)
        #print(model.predict(img_c.reshape(1, 400, order='F')))
        p=my_utils.myPredict(model,img_c.reshape(1, 400, order='F'))
        pred=np.where(p==np.max(p))
        print("当前手写图案预测为数字：", pred[1][0])
        '''
        p = mulClassLogical.myPredict(all_theta,
                                      img_c.reshape(1, 400, order='F'))  # 特别注意这里,因为样本是默认以列方向reshape的，而python里默认按行
        pred = int(p[0, 0])
        if pred == 10:
            pred = 0
        print("当前手写图案预测为数字：", pred)
        '''
        print(tip)
    elif k == ord('e'):
        X, y = my_utils.selOneData(trainSetFile)  #随机从样本集中抽取一行
        X = X.reshape(1, -1)
        y = y.reshape(1, -1)
        # print(X)
        # print(X.shape, y.shape)
        # X = (X + 1) / 2
        img_c = X * 255  # 对样本像素点数值还原
        # print(X)
        img_c = cv2.resize(img_c.reshape(
            20, 20, order='F'), (c_width, c_height))
        img = img_c  # 更新显示的图像
        isGrayMode = True
        # print(img[100])
        pred = int(y[0, 0])
        if pred == 10:
            pred = 0
        print("当前样本图案数字标签为：", pred)
        p=my_utils.myPredict(model,X,needNorm=False)
        pred=np.where(p==np.max(p))
        print("当前样本图案预测数字为：", pred[1][0])
        '''
        p = mulClassLogical.myPredict(all_theta, X)
        pred = int(p[0, 0])
        if pred == 10:
            pred = 0
        print("当前样本图案预测数字为：", pred)
        '''
        print(tip)
    elif k == 27:  # 27为esc键
        break

cv2.destroyAllWindows()
