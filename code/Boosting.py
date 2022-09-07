import os
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from Data_boosting import DATA
from keras.models import Sequential,Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout,Input,Multiply,LocallyConnected1D,Reshape,Add
import keras
from keras_applications.resnet50 import ResNet50
from keras_applications.xception import Xception
from keras_applications.mobilenet import MobileNet
import tensorflow as tf
import keras as K
from keras.initializers import Constant
from keras.engine.topology import Layer
import cv2

'''
用于将alpha和其他类分开
'''

def multi_category_focal_loss1(y_true, y_pred):
    epsilon = 1.e-7
    gamma = 2.0
    alpha = tf.constant([[0.5],[0.3],[0.5]], dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.math.log(y_t)
    weight = tf.math.pow(tf.subtract(1., y_t), gamma)
    fl = tf.matmul(tf.multiply(weight, ce), alpha)
    loss = tf.reduce_mean(fl)
    return loss



class Mobile_MODEL_CJ_old(object):
    def __init__(self):
        ppath = os.getcwd().split('/')
        real_path = ppath[0]
        for k in range(len(ppath) - 1):
            real_path = real_path + ppath[k] + '/'

        check_point = real_path + 'user_data/fineturned_model/MobileNet_CJ.hdf5'
        self.model = K.models.load_model(check_point, custom_objects={'multi_category_focal_loss1':multi_category_focal_loss1})

    def find_idx(self, label):
        mmax = np.max(label)
        for k in range(3):
            if label[k] == mmax:
                return k


    def noramlization(self, data):  
        #归一化，放缩到0-1之间
        minVals = np.min(data)
        maxVals = np.max(data)
        normData = (data - minVals) / (maxVals - minVals + 0.0001)
        return normData


    def preprocess(self, data):
        '''
        加上旋转,变换
        ''' 
        data = data[0]
        if data.shape[0] > data.shape[1]:
            new_data = np.zeros((data.shape[1],data.shape[0], data.shape[2]))
            for kk in range(data.shape[2]):
                new_data[:,:,kk] = data[:,:,kk].T
            data = new_data.copy()
        data = cv2.resize(data, (320, 160))
        data_0 = data[:,:,0].copy()
        data_1 = data[:,:,1].copy()
        data[:,:,1] = data_0
        data[:,:,0] = data_1

        data_lis = [None] * 4


            
        for k in range(4):
            new_data = data.copy()
            flag = k - 1
            if k!=2:
                new_data = cv2.flip(new_data, flag)

            data_lis[k] = new_data.copy()

        data = np.array(data_lis)
        return data

    def predict(self,x):
        x = self.preprocess(x)
        predict_lis = [None] * 4
        for k in range(4):
            predict_lis[k] = self.model.predict(np.array([x[k]]))[0]
        predict_label1 = np.array(predict_lis)
        predict_label1 = np.mean(predict_label1, axis = 0)
        predict_label = self.find_idx(predict_label1)
        return predict_label, predict_label1


class Moblie_MODEL(object):
    def __init__(self):
        ppath = os.getcwd().split('/')
        real_path = ppath[0]
        for k in range(len(ppath) - 1):
            real_path = real_path + ppath[k] + '/'
        check_point = real_path + 'user_data/fineturned_model/Moblie_85.hdf5'

        self.model = K.models.load_model(check_point, custom_objects={'multi_category_focal_loss1':multi_category_focal_loss1})
        self.data_conf = {}
        self.data_conf['BATCH_SIZE'] = 8             #一个batch
        self.data_conf['SHAPE'] = [160,320]          #图片统一大小,若不resize 使用[-1,-1]
        self.data_conf['IS_FLIP'] = False             #是否通过翻转增加数据集
        self.data_conf['PRE_PROCESS_METHOD'] = 0     #0:不做处理,1:标准化,2:缩放至0-1之间
        self.data_conf['CHANNEL_SIZE'] = 2           #0:只用mag,1:只用con,2:con和mag一起用



    def find_idx(self, label):
        mmax = np.max(label)
        for k in range(3):
            if label[k] == mmax:
                return k


    def preprocess(self, data):
        '''
        用于将1*h*w*c的图片处理成该有的样子
        使用preprocess文件夹中的数据
        resize成128*256即可
        '''
        #resize
        data = data[0]
        if data.shape[0] > data.shape[1]:
            new_data = np.zeros((data.shape[1],data.shape[0], data.shape[2]))
            for kk in range(data.shape[2]):
                new_data[:,:,kk] = data[:,:,kk].T
            data = new_data.copy()
        #print(data.shape)
        data = cv2.resize(data, (320, 160))

        data = np.array([data])
        return data

    def predict(self,x):
        x = self.preprocess(x)
        predict_label1 = self.model.predict(x)
        predict_label = self.find_idx(predict_label1[0])
        return predict_label, predict_label1



class Moblie_CE_MODEL(object):
    def __init__(self):
        ppath = os.getcwd().split('/')
        real_path = ppath[0]
        for k in range(len(ppath) - 1):
            real_path = real_path + ppath[k] + '/'

        check_point = real_path + 'user_data/fineturned_model/Mobile_CE_83.hdf5'
        self.model = K.models.load_model(check_point)


    def find_idx(self, label):
        mmax = np.max(label)
        for k in range(3):
            if label[k] == mmax:
                return k


    def noramlization(self, data):  
        #归一化，放缩到0-1之间
        minVals = np.min(data)
        maxVals = np.max(data)
        normData = (data - minVals) / (maxVals - minVals + 0.0001)
        return normData


    def preprocess(self, data):
        '''
        用于将1*h*w*c的图片处理成该有的样子
        使用preprocess文件夹中的数据
        resize成128*256即可
        '''
        #resize
        data = data[0]
        if data.shape[0] > data.shape[1]:
            new_data = np.zeros((data.shape[1],data.shape[0], data.shape[2]))
            for kk in range(data.shape[2]):
                new_data[:,:,kk] = data[:,:,kk].T
            data = new_data.copy()
        #print(data.shape)
        data = cv2.resize(data, (256, 128))
        data[:,:,0] = self.noramlization(data[:,:,0])
        data[:,:,1] = self.noramlization(data[:,:,1])
        data = np.array([data])
        return data

    def predict(self,x):
        x = self.preprocess(x)
        predict_label1 = self.model.predict(x)
        predict_label = self.find_idx(predict_label1[0])
        return predict_label, predict_label1





class Boosting(object):
    def __init__(self):
        self.moblie_model = Moblie_MODEL()
        self.moblie_ce_model = Moblie_CE_MODEL()
        self.moblie_model_cj = Mobile_MODEL_CJ_old()




        conf_dict = {}
        conf_dict['LABEL_LIS'] = ['alpha', 'beta', 'betax']
        ppath = os.getcwd().split('/')
        real_path = ppath[0]
        for k in range(len(ppath) - 1):
            real_path = real_path + ppath[k] + '/'
        conf_dict['DATA_DIR_PATH'] = real_path + 'user_data/valide_zxb/'
        self.data_origin = DATA()
        self.data_origin.update_settings(conf_dict)


        self.boosting_lis = [1,1,1]

    def find_idx(self, label):
        mmax = np.max(label)
        for k in range(3):
            if label[k] == mmax:
                return k



    def run(self, cj_label=None):
        idx = 0
        result_lis = [None] * 40000
        iidx = 0

        if cj_label is None:
            cj_label = np.load('cj_label.npy')

        for k in range(len(self.data_origin.data_path_lis)):
            print(k)
            x,y = self.data_origin.get_single_data(k)
            label_moblie, label_moblie_p = self.moblie_model.predict(x.copy())
            label_moblie_ce, label_moblie_ce_p = self.moblie_ce_model.predict(x.copy())
            label_moblie_cj, label_moblie_cj_p = self.moblie_model_cj.predict(x.copy())
            label_p = self.boosting_lis[0] * label_moblie_ce_p + self.boosting_lis[1] * label_moblie_p + self.boosting_lis[2]*label_moblie_cj_p
            predict_label = self.find_idx(label_p[0])

            if predict_label != 0:
                name = self.data_origin.data_path_lis[k][0].split('/')[-1][:-6]
                name = int(name)
                label_p = cj_label[name-1][0]
                predict_label = self.find_idx(label_p)


            name = self.data_origin.data_path_lis[k][0].split('/')[-1][:-6]
            result_lis[iidx] = [name, predict_label + 1]
            iidx = iidx + 1
        
        ppath = os.getcwd().split('/')
        real_path = ppath[0]
        for k in range(len(ppath) - 1):
            real_path = real_path + ppath[k] + '/'

        txt_path = real_path + 'classification_result/result.txt'
        f = open(txt_path, 'w')
        result_lis = result_lis[:iidx]
        for k in range(iidx):
            f.write(result_lis[k][0] + ' ' + str(result_lis[k][1]) + '\n')

if __name__ == '__main__':
    bt = Boosting()
    bt.run()