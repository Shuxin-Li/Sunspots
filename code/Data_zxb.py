import os
import numpy as np
from skimage import io
from skimage.transform import resize

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

class Data(object):
    def __init__(self):
        self.test_data = None

    def load_data(self):
        ppath = os.getcwd().split('/')
        real_path = ppath[0]
        for k in range(len(ppath) - 1):
            real_path = real_path + ppath[k] + '/'

        check_point = real_path + 'user_data/input_data.npy'
        self.test_data = np.load(check_point)
      
if __name__ == '__main__':
    data = Data()
    data.load_data()
    print(data.test_data.shape)