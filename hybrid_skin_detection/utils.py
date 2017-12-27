import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import colorsys

BATCH_SIZE = 4

def get_im(path, img_rows, img_cols):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_rows, img_cols))
    img_reshape = np.reshape(img, (img.shape[0]*img.shape[1], 1, img.shape[2]))
    img_squeeze = np.squeeze(img_reshape)

    img_gray = cv2.imread(path, 0)
    img_gray = cv2.resize(img_gray, (img_rows, img_cols))
    img_gray_reshape = np.reshape(img_gray, (img_gray.shape[0]*img_gray.shape[1], 1))
    img_gray_squeeze = np.squeeze(img_gray_reshape)
    #return img_squeeze/255.0, img_gray_squeeze/255.0
    return img/255.0, img_gray/255.0

def _get_deviation(batch_dict):
	mean = np.mean(list(batch_dict.elements()));
	dev = 0.0;
	for key in batch_dict:
		dev += (key-mean)**2*(float(batch_dict[key])/BATCH_SIZE)
	return dev+1e-6

def _get_uniformity(batch_dict):
	uni = 0.0
	for key in batch_dict:
		uni += (float(batch_dict[key])/BATCH_SIZE)**2;
	return uni;

def _get_smoothness(batch_dict):
	return 1-1.0/(1+_get_deviation(batch_dict))

def _get_entropy(batch_dict):
	ent = 0.0;
	for key in batch_dict:
		ent += (float(batch_dict[key])/BATCH_SIZE)*np.log2(float(batch_dict[key])/BATCH_SIZE)
	return ent

def _get_skewness(batch_dict):
	mean = np.mean(list(batch_dict.elements()));
	int_list = [key-mean for key in batch_dict.keys()]
	return (np.mean(int_list)/_get_deviation(batch_dict))**3

def _get_kurtosis(batch_dict):
	mean = np.mean(list(batch_dict.elements()));
	int_list = [key-mean for key in batch_dict.keys()]
	return (np.mean(int_list)/_get_deviation(batch_dict))**4

def _get_properties(batch_dict):
	l = len(list(batch_dict.elements()))
	prop_list = []
	prop_list.append([_get_deviation(batch_dict)]*l);
	prop_list.append([_get_uniformity(batch_dict)]*l)
	prop_list.append([_get_smoothness(batch_dict)]*l)
	prop_list.append([_get_entropy(batch_dict)]*l)
	prop_list.append([_get_skewness(batch_dict)]*l)
	prop_list.append([_get_kurtosis(batch_dict)]*l)
	return np.array(prop_list)

def rgb_yiq(batch):
	for i in range(batch.shape[0]):
		batch[i, :] = np.array(colorsys.rgb_to_yiq(batch[i,0], batch[i,1], batch[i,2]))
	return batch

def gene_matrix(path, img_rows, img_cols, size=BATCH_SIZE, color='rgb'):
	BATCH_SIZE=size
	img, img_gray = get_im(path, img_rows, img_cols)   # img (224,224, 3), img_gray (224,224)
	#num = len(img_gray)//BATCH_SIZE
	for r_i in range(int(np.ceil(img_rows/float(BATCH_SIZE)))):
		for c_i in range(int(np.ceil(img_cols/float(BATCH_SIZE)))):
			r_start = r_i*BATCH_SIZE;
			c_start = c_i*BATCH_SIZE;
			img_batch = img[r_start:r_start+BATCH_SIZE, c_start:c_start+BATCH_SIZE,:]
			img_gray_batch = img_gray[r_start:r_start+BATCH_SIZE, c_start:c_start+BATCH_SIZE];

			img_batch_reshape = np.reshape(img_batch, (img_batch.shape[0]*img_batch.shape[1], img_batch.shape[2]))
			if color=='yiq':
				color_matrix = rgb_yiq(img_batch_reshape)
			else:
				color_matrix = img_batch_reshape
			color_matrix = color_matrix.T   # color_matrix (3, BATCH_SIZE)

			img_gray_reshape = np.reshape(img_gray_batch, (img_gray_batch.shape[0]*img_gray_batch.shape[1], 1))
			img_gray_reshape = np.squeeze(img_gray_reshape)
			batch_dict = Counter(img_gray_reshape)
			prop_matrix = _get_properties(batch_dict);

			batch_matrix = np.concatenate((color_matrix, prop_matrix), axis=0)  # batch_matrix  (9, BATCH_SIZE)

			if r_i==0 and c_i==0:
				img_matrix = batch_matrix;
			else:
				img_matrix = np.concatenate((img_matrix, batch_matrix), axis=1);  # img_matrix  (9, BATCH_SIZE*n)

	return img_matrix


#img_matrix = gene_matrix('/Users/wangyiming/Documents/17Fall/ECE544/project/skin detection/1.jpg', 224, 224)
#print (img_matrix.shape)
#print (img_matrix[0,:])



