import os
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from Data_zxb import Data
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
import random



def multi_category_focal_loss1(y_true, y_pred):
    epsilon = 1.e-7
    gamma = 2.0
    alpha = tf.constant([[0.5],[0.3],[0.5]], dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.math.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.matmul(tf.multiply(weight, ce), alpha)
    loss = tf.reduce_mean(fl)
    return loss

def binary_focal_loss_fixed(y_true, y_pred):
    alpha = 0.5
    gamma = 2
    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true*alpha + (K.backend.ones_like(y_true)-y_true)*(1-alpha)
    
    p_t = y_true*y_pred + (K.backend.ones_like(y_true)-y_true)*(K.backend.ones_like(y_true)-y_pred) + K.backend.epsilon()
    focal_loss = - alpha_t * K.backend.pow((K.backend.ones_like(y_true)-p_t),gamma) * K.backend.log(p_t)
    return K.backend.mean(focal_loss)



class Xception_MODEL_CJ_old(object):
    def __init__(self):
        ppath = os.getcwd().split('/')
        real_path = ppath[0]
        for k in range(len(ppath) - 1):
            real_path = real_path + ppath[k] + '/'
        check_point = real_path + 'user_data/fineturned_model/Xception_CJ.hdf5'
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




class Boosting(object):
    def __init__(self):
        self.xecption_model_cj_old = Xception_MODEL_CJ_old()
        
        self.data = Data()
        self.data.load_data()


    
    def find_idx(self, label):
        mmax = np.max(label)
        for k in range(3):
            if label[k] == mmax:
                return k



    def run(self, is_save = 0):
        self.eval_lis = np.zeros((3,4))
        false_predict_lis_FN = np.zeros((3,1))
        false_predict_lis_FP = np.zeros((3,1))

        acc = 0
        skip = 0
        skip_lis = np.zeros(3)

        X = [None] * self.data.test_data.shape[0]
        iidx = 0
        for k in range(self.data.test_data.shape[0]):
            print(k)
            x = np.array([self.data.test_data[k]])


            label_xception_cj_old, label_xception_cj_old_p = self.xecption_model_cj_old.predict(x)
            X[k] = [label_xception_cj_old_p]
            print(k, label_xception_cj_old)
        X = np.array(X)
        
        if is_save == 1:
            np.save('cj_label', X)
        return X


if __name__ == '__main__':
    bt = Boosting()
    bt.run(is_save=1)