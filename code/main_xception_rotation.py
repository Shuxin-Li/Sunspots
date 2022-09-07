# 黑子分类主文件

import os
import numpy as np
import tensorflow.keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from data_xception import *
# from data_MobileNet import *
#from module import *
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import pandas as pd
from sklearn.metrics import f1_score
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)]
# )


def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

input_shape = (160, 160, 1)
# mask = np.zeros((320, 320))
# z = np.ones((160, 320))
# mask[80:240, :] = z
# print(mask)
train_generator = img_generator(mode='train', bitch_size=4,
                                resize_shape=input_shape,
                                input_type='magnetogram')
val_generator = img_generator(mode='validation', bitch_size=1,
                              resize_shape=input_shape,
                              input_type='magnetogram')
# model = ChiNet(Input_shape=input_shape)
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(3, activation='softmax'))

model = Xception(weights=None,
                 input_shape=input_shape,
                 classes=3)

model.summary()
model.compile(optimizer=keras.optimizers.Adadelta(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

reduce_lr = LearningRateScheduler(scheduler)

model_checkpoint = ModelCheckpoint('D:/lsx/lsx/lsx/tianchi/new_imgaes/Xception_{epoch:04d}_{val_loss:.04f}_{val_accuracy:.04f}.hdf5',
                                   monitor='val_accuracy', verbose=1)  # 保存权重路径
# # 校验集上最优的模型
#model.load_weights('D:/lsx/lsx/lsx/tianchi/new_imgaes/Xception_0013_0.4536_0.6900.hdf5')

history = model.fit_generator(generator=train_generator.__iter__(), steps_per_epoch=1000, epochs=100, verbose=1,
                              callbacks=[model_checkpoint, reduce_lr],
                              validation_data=val_generator.__iter__(), validation_steps=1000, initial_epoch=13)

[val_x, val_y] = generate_val_data(resize_shape=input_shape, sample_range=0.8)
# val_x = val_x[0:2,:,:,:]
predicted_label = model.predict(val_x)

##  Test Time Augmentation, reference https://note.nkmk.me/en/python-numpy-flip-flipud-fliplr/
TTA = 'True'
if TTA == 'True':
    val_x1 = val_x[:, ::-1, :, :]
    val_x2 = val_x[:, :, ::-1, :]
    val_x3 = val_x[:, ::-1, ::-1, :]
    predicted_label1 = model.predict(val_x1)
    predicted_label2 = model.predict(val_x2)
    predicted_label3 = model.predict(val_x3)
    predicted_label_TTA = predicted_label + predicted_label1 + predicted_label2 + predicted_label3
    predicted_label_TTA4 = predicted_label_TTA/4

    predicted_label_TTA8 = np.zeros((1, 3))

    for i in range(8):
        turn_image = TTT8(val_x.copy())
        predicted_label_i = model.predict(turn_image)
        # print(predicted_label_i.shape)
        predicted_label_TTA8 = predicted_label_TTA8 + predicted_label_i

    predicted_label_TTA8 = predicted_label_TTA8/8
    predicted_label_TTA8 = predicted_label_TTA4 + 0.5*predicted_label_TTA8
    # print(predicted_label_TTA4)
    # print(predicted_label_TTA8)


predicted_label = pd.DataFrame(predicted_label)
predicted_label_TTA4 = pd.DataFrame(predicted_label_TTA4)
predicted_label_TTA8 = pd.DataFrame(predicted_label_TTA8)

predicted_label = predicted_label.idxmax(axis=1) + 1
predicted_label_TTA4 = predicted_label_TTA4.idxmax(axis=1) + 1  # 取概率最大的作为label, axis=1表示按行取最大值
predicted_label_TTA8 = predicted_label_TTA8.idxmax(axis=1) + 1

### 计算f1
f1_scores = []
f1_scores_TTA4 = []
f1_scores_TTA8 = []

for i in range(1, 4):
    binary_label = val_y == i
    binary_predicted = predicted_label == i
    f1_scores.append(f1_score(binary_label, binary_predicted, pos_label=1, average='binary'))

for i in range(1, 4):
    binary_label = val_y == i
    binary_predicted_TTA = predicted_label_TTA4 == i
    f1_scores_TTA4.append(f1_score(binary_label, binary_predicted_TTA, pos_label=1, average='binary'))

for i in range(1, 4):
    binary_label = val_y == i
    binary_predicted_TTA = predicted_label_TTA8 == i
    f1_scores_TTA8.append(f1_score(binary_label, binary_predicted_TTA, pos_label=1, average='binary'))

print(f1_scores)
print(f1_scores_TTA4)
print(f1_scores_TTA8)






# 测试集的路径
save_path1 = r'D:\lsx\lsx\lsx\tianchi\images_test\images_test\continuum'
save_path2 = r'D:\lsx\lsx\lsx\tianchi\images_test\images_test\magnetogram'
# save_path1 = r'C:\Users\85048\Desktop\CME\test_input\dataset\continuum\continuum'
# save_path2 = r'C:\Users\85048\Desktop\CME\test_input\dataset\magnetogram\magnetogram'


# read_excel('real_test(2).xls')
# label1 = np.load('label.npy')
read_excel('D:/lsx/lsx/lsx/tianchi/赛道一.test_output_all.xls')
label = np.load('D:/lsx/lsx/lsx/tianchi/赛道一.test_output_all.npy')
print(label)
# print(label[0])
label = pd.DataFrame(label)
# print(label1.T[0])

# file_name = '赛道一-test_output_all.txt'
# label = []
# file = open(file_name, mode='r')
# for line in file:
#     line = line.split()
#     if line[1] == 'alpha':
#         line[1] = 1
#     elif line[1] == 'beta':
#         line[1] = 2
#     else:
#         line[1] = 3
#     label.append(line[1])
# file.close()
# # label = np.array(label)
# label = pd.DataFrame(label)
# print(label)


data1, data2 = [], []
for name1, name2 in zip(os.listdir(save_path1), os.listdir(save_path2)):
    filename1 = os.path.join(save_path1, name1)
    filename2 = os.path.join(save_path2, name2)
    image1 = read_png(filename1, resize_shape=input_shape)
    data1.append(image1)
    image2 = read_png(filename2, resize_shape=input_shape)
    data2.append(image2)

input1 = np.array(data1)
input2 = np.array(data2)
test_data = np.concatenate((input1, input2), axis=-1)
# test_data = test_data[0:2, :, :, :]

# 预测
predicted_label_test = model.predict(test_data)

##  Test Time Augmentation, reference https://note.nkmk.me/en/python-numpy-flip-flipud-fliplr/
TTA = 'True'
if TTA == 'True':
    test_data1 = test_data[:, ::-1, :, :]
    test_data2 = test_data[:, :, ::-1, :]
    test_data3 = test_data[:, ::-1, ::-1, :]
    predicted_label1_test = model.predict(test_data1)
    predicted_label2_test = model.predict(test_data2)
    predicted_label3_test = model.predict(test_data3)
    predicted_label_TTA_test = predicted_label_test + predicted_label1_test + predicted_label2_test + predicted_label3_test
    predicted_label_TTA4_test = predicted_label_TTA_test/4

    predicted_label_TTA8_test = np.zeros((1, 3))

    for i in range(8):
        turn_image = TTT8(test_data.copy())
        predicted_label_i_test = model.predict(turn_image)
        # print(predicted_label_i_test.shape)
        predicted_label_TTA8_test = predicted_label_TTA8_test + predicted_label_i_test

    predicted_label_TTA8_test = predicted_label_TTA8_test/8
    predicted_label_TTA8_test = predicted_label_TTA4_test + 0.5*predicted_label_TTA8_test
    # print(predicted_label_TTA4_test)
    # print(predicted_label_TTA8_test)


predicted_label_test = pd.DataFrame(predicted_label_test)
predicted_label_TTA4_test = pd.DataFrame(predicted_label_TTA4_test)
predicted_label_TTA8_test = pd.DataFrame(predicted_label_TTA8_test)

predicted_label_test = predicted_label_test.idxmax(axis=1) + 1    # 取概率最大的作为label, axis=1表示按行取最大值
predicted_label_TTA4_test = predicted_label_TTA4_test.idxmax(axis=1) + 1
predicted_label_TTA8_test = predicted_label_TTA8_test.idxmax(axis=1) + 1

# print(label.shape)
# print(predicted_label_test.shape)

### 计算f1
f1_scores = []
f1_scores_TTA4 = []
f1_scores_TTA8 = []

for i in range(1, 4):
    binary_label = label == i
    binary_predicted = predicted_label_test == i
    f1_scores.append(f1_score(binary_label, binary_predicted, pos_label=1, average='binary'))

for i in range(1, 4):
    binary_label = label == i
    binary_predicted_TTA = predicted_label_TTA4_test == i
    f1_scores_TTA4.append(f1_score(binary_label, binary_predicted_TTA, pos_label=1, average='binary'))

for i in range(1, 4):
    binary_label = label == i
    binary_predicted_TTA = predicted_label_TTA8_test == i
    f1_scores_TTA8.append(f1_score(binary_label, binary_predicted_TTA, pos_label=1, average='binary'))

print(f1_scores)
print(f1_scores_TTA4)
print(f1_scores_TTA8)

# 创建excel
# predicted = pd.concat([predicted_label_test, predicted_label_test_TTA, label1], axis=1)
# writer = pd.ExcelWriter('C:/Users/85048/Desktop/CME/predicted_test.xlsx')
# predicted.to_excel(writer, 'page_1', float_format='%.10f')
# writer.save()
# writer.close()

# 创建txt
# np.savetxt("C:/Users/85048/Desktop/CME/C_机器智能与学习.txt", predicted_label_test)
# np.savetxt("C:/Users/85048/Desktop/CME/C_机器智能与学习.txt", predicted_label_test_TTA1)

# #单元格上色
# def color_execl(file_name):
#     styleBlueBkg = xlwt.easyxf('pattern: pattern solid, fore_colour red;')  # 红色
#     rb = xlrd.open_workbook(file_name)      #打开t.xls文件
#     ro = rb.sheets()[0]                     #读取表单0
#     wb = copy(rb)                           #利用xlutils.copy下的copy函数复制
#     ws = wb.get_sheet(0)                       #获取表单0
#     for col in range(1,4):                     #循环修改的列
#         for i in range(1,1173):               #循环所有的行
#             if label[i-1] != predicted_label_test1[i-1]:
#                 ws.write(i,col,ro.cell(i, col).value,styleBlueBkg)
#
#     for col in range(4,7):                     #循环修改的列
#         for i in range(1,1173):               #循环所有的行
#             if label[i-1] != predicted_label_test_TTA1[i-1]:
#                 ws.write(i,col,ro.cell(i, col).value,styleBlueBkg)
#     wb.save('C:/Users/85048/Desktop/CME/predicted_test_color.xlsx')
#
#
# if __name__ == '__main__':
#     file_name = 'C:/Users/85048/Desktop/CME/predicted_test.xlsx'
#     color_execl(file_name)