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


#np.random.seed(2016)
use_cache = 0
# color type: 1 - grey, 3 - rgb
color_type_global = 3

# color_type = 1 - gray
# color_type = 3 - RGB
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


def get_im(path, img_rows, img_cols, color_type=1):
    '''
    Load the image from path and reshape it into img_rows*img_cols
    '''

    # Load as grayscale is color_type=1
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    
    resized = cv2.resize(img, (img_cols, img_rows))
    mean_pixel = [103.939, 116.799, 123.68]
    resized = resized.astype(np.float32, copy=False)
    resized-=mean_pixel[0]
    return resized


def get_driver_data():
    '''
    Get the driver information from the csv file. Store the relation between class and images
    '''
    dr = dict()
    path = os.path.join('.', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr


def load_train(img_rows, img_cols, color_type=1):
    '''
    Load train dataset fromt the directory
    '''

    X_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    xx = 0
    for j in range(10):
        print('Load folder c{}'.format(j))
        #path = os.path.join('.', 'imgs', 'train',
        #                    'c' + str(j), '*.jpg')
        path = 'imgs_raw/train/c'+str(j)+'/*.jpg'
        files = glob.glob(path)
        for fl in files[1700:]:
            flbase = os.path.basename(fl)
            img = get_im(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])
            xx += 1

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers

def load_test(img_rows, img_cols, color_type=1):
    '''
    Load test data from directory
    '''
    print('Read test images')
    #path = os.path.join('.', 'imgs', 'test', '*.jpg')
    path = 'imgs_raw/test/*.jpg';
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    thr = math.floor(len(files)/10)
    for i, fl in enumerate(files):
        if i<thr*7:
            continue;
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        if len(X_test) % thr == 0:
            print('Read {} images from {}'.format(len(X_test), len(files)))

    return X_test, X_test_id

def save_model(model, index, cross=''):
    '''
    Save model to file
    '''
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)

def read_model(index, cross=''):
    '''
    Read model from file
    '''
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model


def create_submission(predictions, test_id, info):
    '''
    Store the prediction output for test data into certain csv format
    '''
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

def read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                              color_type=1):
    '''
    Read and mange training data
    '''
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')

    train_data, train_target, driver_id, unique_drivers = \
        load_train(img_rows, img_cols, color_type)
    print ('train data matrix has shape '+ str(len(train_data))+' '+str(len(train_data[0])))
    
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], color_type,
                                        img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))

    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    #for c in range(3):
    #    train_data[:, c, :, :] = train_data[:, c, :, :] - mean_pixel[c]
    train_data[:, :, :]-=mean_pixel[0]
    # train_data /= 255
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, driver_id, unique_drivers

def read_and_normalize_test_data(img_rows=224, img_cols=224, color_type=1):
    '''
    Read and manage test data
    '''
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')
    
    test_data, test_id = load_test(img_rows, img_cols, color_type)
    
    test_data = np.array(test_data, dtype=np.uint8)

    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], color_type,
                                      img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))
    
    test_data = test_data.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]
    # test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id

def dict_to_list(d):
    '''
    Change the data from dictionary to list
    '''
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret

def merge_several_folds_mean(data, nfolds):
    '''
    Merge the result of several models using the mean of outputs
    '''

    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def merge_several_folds_geom(data, nfolds):
    '''
    Merge the result of several models using the geom of outputs
    '''
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1/nfolds)
    return a.tolist()


def CNN_Model_Design(img_rows, img_cols, color_type):
    '''
    VGG16 model implementation
    '''
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(img_rows, img_cols, color_type)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.summary()

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy',
    #          optimizer='adadelta',
    #          metrics=['accuracy'])
    
    return model;

def AlexNet(img_rows, img_cols, color_type):
    '''
    Alexnet model implementation
    '''
    model = Sequential()

    # Conv layer 1 output shape (55, 55, 48)
    model.add(Conv2D(
        kernel_size=(11, 11), 
        data_format="channels_last", 
        activation="relu",
        filters=48, 
        strides=(4, 4), 
        input_shape=(img_rows, img_cols, color_type)
    ))
    model.add(Dropout(0.25))

    # Conv layer 2 output shape (27, 27, 128)
    model.add(Conv2D(
        strides=(2, 2), 
        kernel_size=(5, 5), 
        activation="relu", 
        filters=128
    ))
    model.add(Dropout(0.25))

    # Conv layer 3 output shape (13, 13, 192)
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu", 
        filters=192,
        padding="same",
        strides=(2, 2)
    ))
    model.add(Dropout(0.25))

    # Conv layer 4 output shape (13, 13, 192)
    model.add(Conv2D(
        padding="same", 
        activation="relu",
        kernel_size=(3, 3),
        filters=192
    ))
    model.add(Dropout(0.25))

    # Conv layer 5 output shape (128, 13, 13)
    model.add(Conv2D(
        padding="same",
        activation="relu", 
        kernel_size=(3, 3),
        filters=128
    ))
    model.add(Dropout(0.25))

    # fully connected layer 1
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.25))

    # fully connected layer 2
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.25))

    # output
    model.add(Dense(10, activation='softmax'))

    # optimizer=SGD
    sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model


def VGG_keras(img_rows, img_cols, color_type):
    '''
    ResNet model implementation
    '''
    model_resnet_conv = keras.applications.resnet50.ResNet50(weights=None, classes=10)
    
    keras_input = Input(shape=(img_rows, img_cols, color_type), name = 'image_input')
    output_resnet_conv = model_resnet_conv(keras_input)
    
    pretrained_model = Model(inputs=keras_input, outputs=output_resnet_conv)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    pretrained_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return pretrained_model



def run_cross_validation(nfolds=10, nb_epoch=10, split=0.2, modelStr=''):
    '''
    Training process
    '''

    # Now it loads color image
    # input image dimensions
    #img_rows, img_cols = 240, 280
    img_rows, img_cols = 224, 224
    batch_size = 32
    random_state = 20

    train_data, train_target, driver_id, unique_drivers = \
        read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                                  color_type_global)
    train_data = np.transpose(train_data, (0,2,3,1))
    #train_data = np.transpose(train_data, (0, ))
    print (train_data.shape, train_target.shape)
        
    model = CNN_Model_Design(img_rows, img_cols, color_type_global)
    #model = VGG_keras(img_rows, img_cols, color_type_global)
    #model = AlexNet(img_rows, img_cols, color_type_global)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(train_data, train_target, batch_size=batch_size,
              nb_epoch=nb_epoch,
              verbose=1,
              validation_split=split, shuffle=True, callbacks=[earlyStopping])

    model_json = model.to_json()
    with open("model_resnet_4.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_resnet_4.h5")
    with open('train_history_resnet_4.json', 'w') as fp:
    	json.dump(history.history, fp)
    print("Saved model to disk")
    
    return model;


def test_model_and_submit(start=1, end=1, modelStr=''):
    '''
    Testing process
    '''
    
    img_rows, img_cols = 240, 240
    # batch_size = 64
    # random_state = 51
    nb_epoch = 15

    print('Start testing............')
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols,
                                                      color_type_global)
    yfull_test = []

    for index in range(start, end + 1):
        # Store test predictions
        model = read_model(index, modelStr)
        test_prediction = model.predict(test_data, batch_size=128, verbose=1)
        yfull_test.append(test_prediction)

    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(end - start + 1) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, end - start + 1)
    create_submission(test_res, test_id, info_string)

model = run_cross_validation(2, 200, 0.15, 'CNN_Model_VGG')
