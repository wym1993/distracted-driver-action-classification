import cv2
import numpy as np
import os
from collections import defaultdict
import imageio

driver_store = defaultdict(list)
f = open('./driver_imgs_list.csv', 'r')
for line in f.readlines()[1:]:
	line_split = line.split(',')
	driver_store[line_split[0]].append((line_split[1], line_split[2]))

for driver_id in driver_store.keys():
	img_list = []
	for folder, img_name in driver_store[driver_id]:
		path = 'imgs_raw/train/'+folder+'/'+img_name[:-1]
		print path
		img = imageio.imread(path)
		img = cv2.resize(img, (170, 170))
		img_list.append(img)

	imageio.mimsave(driver_id+'.gif', img_list);
	print 'saved '+driver_id+' to file'