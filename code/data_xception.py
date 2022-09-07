import os
from astropy.io import fits
import numpy as np
from skimage.transform import resize
import random
import keras
import warnings
from skimage import io
from torch import nn
from sklearn import preprocessing
import cv2
import xlrd
import tensorflow as tf
import tensorflow.keras.backend as K

warnings.filterwarnings('ignore')


def noramlization_v1(data):  # 归一化，将样本点的的数值减去最小值，再除以样本点最大于最小数值的差

    minVals = -800
    maxVals = 800
    # normData = 255 * (data - minVals) / (maxVals - minVals)
    data[data > 800] = 800
    data[data < -800] = -800
    normData = 255 * (data - minVals) / (maxVals - minVals)
    normData = normData.astype(np.uint8)
    return normData

def noramlization(data):  # 归一化，将样本点的的数值减去最小值，再除以样本点最大于最小数值的差

    minVals = np.min(data)
    maxVals = np.max(data)
    normData = 255 * (data - minVals) / (maxVals - minVals)
    # normData = (data - minVals) / (maxVals - minVals)
    normData = normData.astype(np.uint8)
    return normData

def noramlization_fits(data, is_mag=0):  # 归一化，将样本点的的数值减去最小值，再除以样本点最大于最小数值的差
    if is_mag == 1:
        data = np.clip(data, -800, 800)
    seq = data.flatten()
    seq.sort()
    # print(seq[0],seq[-1])
    # print(np.mean(seq))
    # # minVals = seq[int(seq.shape[0] * 0.05)]
    # # if seq[0] - minVals < 100:
    minVals = seq[0]

    # # maxVals = seq[int(seq.shape[0] * 0.95)]
    # # if seq[-1] - maxVals < 100:
    maxVals = seq[-1]

    # print(minVals, maxVals)

    normData = 255 * (data - minVals) / (maxVals - minVals)
    normData = normData.astype(np.uint8)
    return normData


# load 进来后保存成npy格式的数据，发现数据太大
def load_data():
    path1 = r'D:\lsx\lsx\lsx\tianchi\trainset\continuum'
    path2 = r'D:\lsx\lsx\lsx\tianchi\trainset\magnetogram'
    save_path1 = r'D:\lsx\lsx\lsx\tianchi\images_test\continuum'
    save_path2 = r'D:\lsx\lsx\lsx\tianchi\images_test\magnetogram'
    if not os.path.exists(save_path1):
        os.mkdir(save_path1)
        os.mkdir(save_path2)

    folds1 = os.listdir(path1)
    folds2 = os.listdir(path2)
    for i, fold1, fold2 in zip(range(len(folds1)), folds1, folds2):
        sub_path1 = os.path.join(path1, fold1)
        sub_path2 = os.path.join(path2, fold2)
        save_path_i_1 = os.path.join(save_path1, fold1)
        sub_path_i_2 = os.path.join(save_path2, fold2)
        sub_fold1 = os.listdir(sub_path1)
        sub_fold2 = os.listdir(sub_path2)
        data1, data2 = [], []
        for name1, name2 in zip(sub_fold1, sub_fold2):
            filename1 = os.path.join(sub_path1, name1)
            filename2 = os.path.join(sub_path2, name2)
            image1 = fits.open(filename1)
            image2 = fits.open(filename2)
            image1.verify('fix')
            image1 = image1[1].data
            image1 = noramlization_fits(image1, is_mag=1)
            save_name1 = os.path.join(save_path_i_1,  name1[0:-5] + '.png')
            io.imsave(save_name1, image1)
            image2.verify('fix')
            image2 = image2[1].data
            image2 = noramlization_fits(image2, is_mag=1)
            save_name2 = os.path.join(sub_path_i_2, str(i) + name1[0:-5] + '.png')
            io.imsave(save_name2, image2)
            # print(image2.shape)
            # data1.append(image1)
            # data2.append(image2)

        # save_name1 = 'data1_' + str(i) + '.npy'
        # save_name2 = 'data2_' + str(i) + '.npy'
        # np.save(save_name1, data1)
        # np.save(save_name2, data2)

def load_numpy(shapes=(748, 1300), resize_shape=(128, 224)):

    data_x1, data_x2, data_y = [], [], []
    for i in range(3):

        save_name1 = 'data1_' + str(i) + '.npy'
        save_name2 = 'data2_' + str(i) + '.npy'
        data1 = np.load(save_name1, allow_pickle=True)
        data2 = np.load(save_name2, allow_pickle=True)
        for x1, x2 in zip(data1, data2):
            shape_x = x1.shape
            if shape_x[0] > shape_x[1]:
                x1 = np.transpose(x1)
                x2 = np.transpose(x2)
                shape_x = x1.shape
            x1 = noramlization(x1)
            x2 = noramlization(x2)
            x1_big = np.pad(x1, (shape_x, shapes))
            x2_big = np.pad(x2, (shape_x, shapes))
            # row_s = int(shapes[0]/2)-int(shape_x[0]/2)
            # col_s = int(shapes[1]/2)-int(shape_x[1]/2)
            # x1_big[row_s:row_s+shape_x[0], col_s:col_s+shape_x[1]] = x1
            # x2_big[row_s:row_s+shape_x[0], col_s:col_s+shape_x[1]] = x2
            x1_big = resize(x1_big, resize_shape)
            x2_big = resize(x2_big, resize_shape)
            data_x1.append(x1_big)
            data_x2.append(x2_big)
            data_y.append(i)

    data_x1 = np.array(data_x1)
    data_x2 = np.array(data_x2)
    data_y = np.array(data_y)
    return data_x1, data_x2, data_y

def read_excel(filename = 'test_real.xlsx'):

    data = xlrd.open_workbook(filename)
    table = data.sheets()[0]
    nrows = table.nrows #行数
    ncols = table.ncols #列数
    label = np.zeros([nrows, ncols-1])
    for i in range(0, nrows):
        for j in range(1, ncols):
            tem_value = table.cell(i, j).value
            label[i, j-1] = tem_value

    np.save('赛道一.test_output_all.npy', label)

def read_fits(filename1, shapes=(748, 1300), resize_shape=(256, 448)):
    image1 = fits.open(filename1)
    image1.verify('fix')
    image1 = image1[1].data
    image1 = noramlization(image1)
    shape_x = image1.shape
    if shape_x[0] > shape_x[1]:
        image1 = np.transpose(image1)
        shape_x = image1.shape
    # image1 = np.pad(image1, (shape_x, shapes), 'constant')
    image1 = resize(image1, resize_shape)
    return image1


def TTT8(data):
    '''
            加上旋转,变换
    '''
    center_lis = [None] * 5
    center_lis[0] = (int(data.shape[2] / 2), int(data.shape[1] / 2))
    center_lis[1] = (0, 0)
    center_lis[2] = (int(data.shape[2]), int(data.shape[1]))
    center_lis[3] = (0, int(data.shape[0]))
    center_lis[4] = (int(data.shape[2]), 0)
    TTA_data = []
    for i in range(data.shape[0]):
        new_data = []
        flag = random.randint(0, 4)
        center = center_lis[flag]
        angle = random.randint(-6, 6)
        scale = random.uniform(0.9, 1.1)
        turn_image = cv2.getRotationMatrix2D(center, angle, scale)  # 旋转矩阵
        # new_data = np.zeros((data.shape[1], data.shape[2], 2))
        # new_data1 = cv2.warpAffine(data[i, :, :, 0], turn_image, (data.shape[2], data.shape[1]))
        # new_data2 = cv2.warpAffine(data[i, :, :, 1], turn_image, (data.shape[2], data.shape[1]))
        # new_data[:,:,0] = new_data1
        # new_data[:,:,1] = new_data2
        new_data = cv2.warpAffine(data[i, :, :, :], turn_image, (data.shape[2], data.shape[1]))
        TTA_data.append(new_data)

    TTA_data = np.array(TTA_data)
    return TTA_data

def read_png(filename1, shapes=(748, 1300), resize_shape=(256, 448)):
    image1 = io.imread(filename1)
    image1 = image1.astype('float32')
    shape_x = image1.shape
    if shape_x[0] > shape_x[1]:
        image1 = np.transpose(image1)
        # shape_x = image1.shape
    # image1 = np.pad(image1, (shape_x, shapes), 'constant')
    # mean = np.mean(image1)
    # std = np.std(image1)
    # image1 -= mean
    # image1 /= std
    image1 = resize(image1, resize_shape[0:-1])
    image1 = np.expand_dims(image1, axis=-1)
    return image1

def multi_category_focal_loss1(y_true, y_pred):
    epsilon = 1.e-7
    gamma = 2.0
    alpha = tf.constant([[0.6], [0.5], [0.7]], dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.math.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.matmul(tf.multiply(weight, ce), alpha)
    loss = tf.reduce_mean(fl)
    return loss

def generate_val_data(resize_shape=(160, 320, 2), sample_range=0.8):

    save_path1 = r'D:\lsx\lsx\lsx\tianchi\images\continuum'
    save_path2 = r'D:\lsx\lsx\lsx\tianchi\images\magnetogram'
    folds = os.listdir(save_path1)
    var_x, var_y = [], []
    for i, fold in zip(range(len(folds)), folds):
        save_path_i_1 = os.path.join(save_path1, fold)
        sub_path_i_2 = os.path.join(save_path2, fold)
        sub_fold1 = os.listdir(save_path_i_1)
        sub_fold2 = os.listdir(sub_path_i_2)
        j = 0
        for name1, name2 in zip(sub_fold1, sub_fold2):
            j = j+1
            if j/len(sub_fold1) > sample_range:
                filename1 = os.path.join(save_path_i_1, name1)
                filename2 = os.path.join(sub_path_i_2, name2)
                data1 = read_png(filename1, resize_shape=resize_shape)
                data2 = read_png(filename2, resize_shape=resize_shape)
                input1 = np.array(data1)
                input2 = np.array(data2)
                data = np.concatenate((input1, input2), axis=-1)
                var_x.append(data)
                var_y.append(i + 1)
    var_x = np.array(var_x)
    var_y = np.array(var_y)
    return var_x, var_y

class img_generator(object):

    def __init__(self, mode='train', bitch_size=16, ratio=(0.325, 0.834),
                        shapes=(748, 1300), resize_shape=(256, 448),
                        img_type='.png', augment='True', input_type='continuum'):
        self.img_type = img_type
        if self.img_type =='.png':
            path1 = r'D:\lsx\lsx\lsx\tianchi\images\continuum'
            path2 = r'D:\lsx\lsx\lsx\tianchi\images\magnetogram'
        elif self.img_type == '.fits':
            path1 = r'D:\lsx\lsx\lsx\tianchi\trainset\continuum'
            path2 = r'D:\lsx\lsx\lsx\tianchi\trainset\magnetogram'
        else:
            raise ValueError('Image type should be .png or fits.')
        self.mode = mode
        self.bitch_size = bitch_size
        self.ratio = ratio
        self.shapes = shapes
        self.resize_shape = resize_shape
        self.sub_path0_1 = os.path.join(path1, 'alpha')
        self.sub_path0_2 = os.path.join(path2, 'alpha')
        self.sub_path0_1_list = os.listdir(self.sub_path0_1)
        self.sub_path0_2_list = os.listdir(self.sub_path0_2)
        self.length0 = len(self.sub_path0_1_list)
        self.sub_path1_1 = os.path.join(path1, 'beta')
        self.sub_path1_2 = os.path.join(path2, 'beta')
        self.sub_path1_1_list = os.listdir(self.sub_path1_1)
        self.sub_path1_2_list = os.listdir(self.sub_path1_2)
        self.length1 = len(self.sub_path1_1_list)
        self.sub_path2_1 = os.path.join(path1, 'betax')
        self.sub_path2_2 = os.path.join(path2, 'betax')
        self.sub_path2_1_list = os.listdir(self.sub_path2_1)
        self.sub_path2_2_list = os.listdir(self.sub_path2_2)
        self.length2 = len(self.sub_path2_1_list)
        self.augment = augment
        self.input_type = input_type

    def __iter__(self):
        if self.mode == 'train':
            sample_range = (0, 0.8)
        elif self.mode == 'validation':
            sample_range = (0.8, 1)
        else:
            sample_range = (0.8, 1)

        batch_list_x1, batch_list_x2, batch_list_y = [], [], []
        while 1:
            sampling = random.random()
            if sampling < self.ratio[0]:
                index = np.random.randint(int(self.length0 * sample_range[0]), int(self.length0 * sample_range[1]))
                filename1 = os.path.join(self.sub_path0_1, self.sub_path0_1_list[index])
                filename2 = os.path.join(self.sub_path0_2, self.sub_path0_2_list[index])
                batch_list_y.append(0)
            elif sampling < self.ratio[1]:
                index = np.random.randint(int(self.length1 * sample_range[0]), int(self.length1 * sample_range[1]))
                filename1 = os.path.join(self.sub_path1_1, self.sub_path1_1_list[index])
                filename2 = os.path.join(self.sub_path1_2, self.sub_path1_2_list[index])
                batch_list_y.append(1)
            else:
                index = np.random.randint(int(self.length2 * sample_range[0]), int(self.length2 * sample_range[1]))
                filename1 = os.path.join(self.sub_path2_1, self.sub_path2_1_list[index])
                filename2 = os.path.join(self.sub_path2_2, self.sub_path2_2_list[index])
                batch_list_y.append(2)
            if self.img_type =='.png':
                image1 = read_png(filename1, self.shapes, self.resize_shape)
                image2 = read_png(filename2, self.shapes, self.resize_shape)

            elif self.img_type =='.fits':
                image1 = read_fits(filename1, self.shapes, self.resize_shape)
                image2 = read_fits(filename2, self.shapes, self.resize_shape)
            else:
                raise ValueError('Image type should be .png or .fits.')

            if self.augment == 'True':
                if random.randint(0, 2) == 0:
                    image1 = np.fliplr(image1)
                    image2 = np.fliplr(image2)
                if random.randint(0, 2) == 0:
                    image1 = np.flipud(image1)
                    image2 = np.flipud(image2)
            batch_list_x1.append(image1)
            batch_list_x2.append(image2)
            if len(batch_list_x1) >= self.bitch_size:
                batch_list_y = np.array(batch_list_y)
                input1 = np.array(batch_list_x1)
                input2 = np.array(batch_list_x2)
                # mask = input1
                # for i in range(mask.shape[0]):
                #     for j in range(mask.shape[1]):
                #         for x in range(mask.shape[2]):
                #             for y in range(mask.shape[3]):
                #                 mask[i][j][x][y] = 1
                output = keras.utils.to_categorical(batch_list_y, 3)

                if self.input_type == 'continuum':
                    # 数据扩充
                    center_lis = [None] * 5
                    center_lis[0] = (int(input1.shape[1] / 2), int(input1.shape[0] / 2))
                    center_lis[1] = (0, 0)
                    center_lis[2] = (int(input1.shape[1]), int(input1.shape[0]))
                    center_lis[3] = (0, int(input1.shape[0]))
                    center_lis[4] = (int(input1.shape[1]), 0)
                    flag = random.randint(0, 4)
                    center = center_lis[flag]  # 随机选取个旋转中心
                    angle = random.randint(0, 20)
                    scale = random.uniform(0.9, 1.1)

                    turn_image = cv2.getRotationMatrix2D(center, angle, scale)  # 旋转矩阵
                    input1 = input1.reshape((input1.shape[0], input1.shape[1], input1.shape[2]))
                    for i in range(input1.shape[0]):
                        input1[i] = cv2.warpAffine(input1[i], turn_image, (input1.shape[1], input1.shape[2]))
                    input1 = input1.reshape((input1.shape[0], input1.shape[1], input1.shape[2], 1))

                    yield (input1, output)
                elif self.input_type == 'magnetogram':
                    # 数据扩充
                    center_lis = [None] * 5
                    center_lis[0] = (int(input1.shape[1] / 2), int(input1.shape[0] / 2))
                    center_lis[1] = (0, 0)
                    center_lis[2] = (int(input1.shape[1]), int(input1.shape[0]))
                    center_lis[3] = (0, int(input1.shape[0]))
                    center_lis[4] = (int(input1.shape[1]), 0)
                    flag = random.randint(0, 4)
                    center = center_lis[flag]  # 随机选取个旋转中心
                    angle = random.randint(0, 20)
                    scale = random.uniform(0.9, 1.1)

                    turn_image = cv2.getRotationMatrix2D(center, angle, scale)  # 旋转矩阵
                    input2 = input2.reshape((input2.shape[0], input2.shape[1], input2.shape[2]))
                    for i in range(input2.shape[0]):
                        input2[i] = cv2.warpAffine(input2[i], turn_image, (input2.shape[1], input2.shape[2]))
                    input2 = input2.reshape((input2.shape[0], input2.shape[1], input2.shape[2], 1))

                    yield (input2, output)
                elif self.input_type == 'both':
                    input3 = np.concatenate((input1, input2), axis=-1)
                    # 数据扩充
                    center_lis = [None] * 5
                    center_lis[0] = (int(input3.shape[1] / 2), int(input3.shape[0] / 2))
                    center_lis[1] = (0, 0)
                    center_lis[2] = (int(input3.shape[1]), int(input3.shape[0]))
                    center_lis[3] = (0, int(input3.shape[0]))
                    center_lis[4] = (int(input3.shape[1]), 0)
                    flag = random.randint(0, 4)
                    center = center_lis[flag]
                    angle = random.randint(0, 20)
                    scale = random.uniform(0.9, 1.1)

                    turn_image = cv2.getRotationMatrix2D(center, angle, scale)  # 旋转矩阵
                    for i in range(input3.shape[0]):
                        input3[i] = cv2.warpAffine(input3[i], turn_image, (input3.shape[1], input3.shape[2]))

                    # print(input3)
                    # print(output)
                    yield (input3, output)
                else:
                    raise ValueError('input_type should be continuum, magnetogram or both.')
                batch_list_x1, batch_list_x2, batch_list_y = [], [], []


if __name__ == '__main__':

    load_data()

    # shapes, resize_shape = (748, 1300), (256, 448)
    # data_x1, data_x2, data_y = load_numpy(shapes=shapes, resize_shape=resize_shape)
    # np.save('shapes.npy', shapes)
    # shapes = np.load('shapes.npy')
    # shapes = np.array(shapes)
    # value = np.max(shapes, 0)
    # print(value)

