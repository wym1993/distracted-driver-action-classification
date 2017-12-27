import numpy as np

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd

file_base = 'knn_pretrain_2_csv/knn_'
li = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

for i in range(1, 10):
	file_path = file_base+str(i)+'.csv'
	df = pd.read_csv(file_path)
	df_new = pd.DataFrame.as_matrix(df[li])
	if i==1:
		mat_all = df_new
	else:
		mat_all+=df_new

mat_all/=9

data = pd.DataFrame(mat_all, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
img_Series = pd.Series(sorted(list(df['img'])), name='img', index=data.index)
res_df = pd.concat([img_Series, data], axis=1)

res_df.to_csv('submission_knn_combined.csv', index=False)
