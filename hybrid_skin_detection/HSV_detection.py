import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt

morph_opening = True
blur = True
kernel_size = 5
iterations = 1

#path_from = '/Users/wangyiming/Documents/17Fall/ECE544/project/imgs/test'
path_from = 'imgs_raw/test'
path_to = 'imgs_new/test'
min_range = np.array([0, 20, 40], dtype='uint8')
max_range = np.array([30, 255, 255], dtype='uint8')

def get_skin(img_path):
	img = cv2.imread(img_path)
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img_filtered = cv2.inRange(img_hsv, min_range, max_range)


	if morph_opening:
		kernel = np.ones((kernel_size,kernel_size), np.uint8)
		img_filtered = cv2.morphologyEx(img_filtered, cv2.MORPH_OPEN, kernel, iterations=iterations)
	a=fig.add_subplot(2,2,3)
	plt.imshow(img_filtered)


	if blur:
		img_filtered = cv2.GaussianBlur(img_filtered, (kernel_size,kernel_size), 0)
	a=fig.add_subplot(2,2,4)
	plt.imshow(img_filtered)

	plt.show()
	return img_filtered;

for i, file in enumerate(os.listdir(path_from)):
	img_filtered = get_skin(path_from+'/'+file);
	cv2.imwrite(path_to+'/'+file, img_filtered)
	print (path_to+'/'+file+' finished')
