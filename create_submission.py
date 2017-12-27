import numpy as np
import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
from numpy.random import permutation
import h5py
import json

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Convolution2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import model_from_json
from keras.callbacks import EarlyStopping

model_num = 4;
path = 'model_vgg_'

def get_im(path, img_rows, img_cols, color_type=1):
    '''
    Read the image from path and change its shape for training
    '''
    
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    #resized = img
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    mean_pixel = [103.939, 116.799, 123.68]
    resized = resized.astype(np.float32, copy=False)
    #print (resized.shape)
    '''
    for c in range(3):
        resized[:, :, c] = resized[:, :, c] - mean_pixel[c]
    '''
    resized-=mean_pixel[0]
    # resized = resized.transpose((2, 0, 1))
    # resized = np.expand_dims(img, axis=0)
    return resized

def load_model(model_path, model_weight):
    '''
    Load model from file
    '''

	json_file = open(model_path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(model_weight)
	print("Loaded model from disk")
	return loaded_model

def load_test(img_rows, img_cols, color_type=1):
    '''
    Load test data from directory
    '''

    print('Read test images')
    #path = os.path.join('.', 'imgs', 'test', '*.jpg')
    path = 'imgs_raw/test/*.jpg';
    #path = '/dev/shm/imgs_new/test/*.jpg'
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    thr = math.floor(len(files)/10)
    for i, fl in enumerate(sorted(files)):
        if i<=60000:
            continue;
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        if len(X_test) % thr == 0:
            print('Read {} images from {}'.format(len(X_test), len(files)))

    return X_test, X_test_id

def read_and_normalize_test_data(img_rows=224, img_cols=224, color_type=1):
    '''
    Reading and managing test data
    '''

    test_data, test_id = load_test(img_rows, img_cols, color_type)
    
    test_data = np.array(test_data, dtype=np.uint8)

    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], color_type,
                                      img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))
    
    test_data = test_data.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    '''
    for c in range(3):
        test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]
    '''
    test_data-=mean_pixel[0]
    # test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id

def merge_several_folds_mean(data, nfolds):
    '''
    Merge the result from several models and use the mean as output
    '''

    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

models = []
for i in range(model_num):
	model_path = path+str(i+1)+'.json'
	model_weight = path+str(i+1)+'.h5';
	models.append(load_model(model_path, model_weight))

img_rows = 224
img_cols = 224
color_type = 3 #RGB
test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type)
test_data = np.transpose(test_data, (0,2,3,1))
predicts = []
for model in models:
	test_prediction = model.predict(test_data, batch_size=32, verbose=1)
	predicts.append(test_prediction);

result = merge_several_folds_mean(predicts, model_num)

result1 = pd.DataFrame(result, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
result1.to_csv('submission_resnet_4.csv', index=False)