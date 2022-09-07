import numpy as np
import os
import cv2
import random

class DATA(object):
    def __init__(self):
        self.conf_dict = {}
        self.data_path_lis = None



    def update_settings(self, conf_dict):
        #配置设置
        self.conf_dict = conf_dict

        #获取全部的数据并装到list里面,并打散
        self.data_path_lis = [None] * 1000000
        idx = 0
        for j in range(len(self.conf_dict['LABEL_LIS'])):
            label = self.conf_dict['LABEL_LIS'][j]
            class_dir_path = self.conf_dict['DATA_DIR_PATH'] + label + '/'

            event_lis = os.listdir(class_dir_path)
            event_lis.sort()
            for k in range(len(event_lis)):
                if k % 2 == 0:
                    mag_file_path = class_dir_path + event_lis[k]
                    con_file_path = mag_file_path[:-5] + '1.png'
                    self.data_path_lis[idx] = [mag_file_path, con_file_path, j]
                    idx = idx + 1
                else:
                    continue
                
        self.data_path_lis = self.data_path_lis[:idx]
        # random.shuffle(self.data_path_lis)
        # print(self.data_path_lis[0])
        # print(self.data_path_lis[-1])
        return



   

    def get_single_data(self,k):
        data_info = self.data_path_lis[k]
        data_mag = cv2.imread(data_info[0], 0)
        data_con = cv2.imread(data_info[1], 0)
        data = np.zeros((data_mag.shape[0], data_mag.shape[1], 2))
        data[:,:,0] = data_mag.copy()
        data[:,:,1] = data_con.copy()
        data = np.array([data])

        label = np.zeros((1, len(self.conf_dict['LABEL_LIS'])))
        label[0][data_info[2]] = 1
        
        return data, label
                          


if __name__ == '__main__':
    conf_dict = {}
    conf_dict['DATA_DIR_PATH'] = '/home/bing/WorkSpace/天池比赛/ZXB_code/dataset_preprocessed/train/'
    conf_dict['LABEL_LIS'] = ['alpha', 'beta', 'betax']
   

    data = DATA()
    data.update_settings(conf_dict)
    X,Y = data.get_single_data(0)
    print(X.shape)
    print(Y.shape)
    