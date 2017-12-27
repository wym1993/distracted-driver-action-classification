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
from sklearn.neighbors import NearestNeighbors

import keras
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, Input,AveragePooling2D, merge, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from sklearn.neighbors import NearestNeighbors

# Load result if necessary
df = pd.read_csv('submission_wu_pretrain_2.csv')
li = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
result = pd.DataFrame.as_matrix(df[li])
test_ids = list(df['img'])
print (result.shape, df.head)

# Load and change size of each image
img_rows, img_cols, color_type = 224, 224, 3
test_images = [];
test_ids = []
resize_row, resize_col = 40, 40
path = os.path.join('.', 'imgs', 'test', '*.jpg')
for i, fl  in enumerate(glob.glob(path)):
    flbase = os.path.basename(fl)
    img = get_im(fl, img_rows, img_cols, color_type)
    img = cv2.resize(img, (resize_row, resize_col))
    test_images.append(img);
    test_ids.append(flbase);
    if i%1001==0:
        print ('read '+str(i)+' pics')
print (len(test_images), test_images[0].shape, len(test_ids))

flatten_test_images = [];
for img in test_images:
    flatten_test_images.append(img.flatten())

neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(flatten_test_images)
print ('train step finished')
print(len(flatten_test_images), len(flatten_test_images[0]))

similar_list = {}
for i, img in enumerate(flatten_test_images):
    similar_list[test_ids[i]] = neigh.kneighbors([img], return_distance=False)
    print (str(i)+'finished')

# Find k most similar images
k = 10;
similar_list = {};
for i, img1 in enumerate(test_images):
    cur_list = [];
    for j, img2 in enumerate(test_images):
        cur_list.append((test_ids[j], np.linalg.norm(img1-img2)))
    cur_list.sort(key=lambda x:x[1]);
    similar_list[test_ids[i]] = [filename for filename, dis in cur_list[:k]];
    print (str(i)+' pic finished')
print (len(similar_list.keys()), similar_list['img_44875.jpg'])

# get the final result based on k most similar images
result_new = []
for idx, img in enumerate(sorted(test_ids)):
    nei_imgs = similar_list[img];
    result_new.append(result[test_ids.index(nei_imgs[9])]) 
    #if idx%1000==0:
    print (str(idx)+' '+img+' imgs have finished')
print (len(result_new), len(result_new[0]))


# Write new result to csv
df = pd.DataFrame(result_new, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
series = pd.Series(sorted(test_ids), name='img')
df = pd.concat([series, df], axis=1)
df.to_csv('knn_pretrain_2_csv/knn_9.csv', index=False)
