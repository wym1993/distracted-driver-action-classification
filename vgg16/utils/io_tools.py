# ECE544 project load train and test data
# fwu11
import os
import pickle
import glob
import cv2

def read_image(path, rows=224, cols=224):
    img = cv2.imread(path)
    resized = cv2.resize(img, (rows, cols))
    return resized


def load_train(data_txt_file, image_data_path):

    print('Read drivers data')

    # get driver data
    f = open(data_txt_file, 'r')
    lines = f.readlines()[1:]
    f.close()

    return lines


def load_test():
    print('Read test images')
    path = os.path.join('.', 'data', 'test', '*.jpg')
    files = glob.glob(path)

    return files

