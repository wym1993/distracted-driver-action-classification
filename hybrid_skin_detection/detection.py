import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

import numpy as np
from utils import gene_matrix
import os
import cv2
import json
import csv

BATCH_SIZE=4
img_row = 224
img_col = 224
pic_num = 5
color='yiq'
train_path = './hybrid_skin_data/train'
pos_neg_thresh = 0.45
min_range = np.array([0, 20, 40], dtype='uint8')
max_range = np.array([30, 255, 255], dtype='uint8')


def build_model(input_dims=9):
	model = Sequential();
	model.add(Dense(32, input_dim=input_dims, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='sigmoid'))

	sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model;

def deal_with_inbalance(x, y):
	pos_idxs = [j for j in range(len(y)) if y[j]==1]
	neg_idxs = [j for j in range(len(y)) if y[j]==0]
	if float(len(pos_idxs))/len(new_y)<pos_neg_thresh:
		store = set();
		while len(store)<len(pos_idxs):
			store.update(np.random.choice(neg_idxs, len(pos_idxs)));
		used_rows = sorted(list(store)+pos_idxs);
	return x[:, used_rows], y[used_rows]

train_arr = [];
for i, img in enumerate(os.listdir(os.path.join(train_path, 'data'))[:pic_num]):
	img_path = os.path.join(train_path, 'data', img)
	img_mat = gene_matrix(img_path, img_row, img_col, size=BATCH_SIZE, color=color)
	train_arr.append(img_mat)
	print i, img_path
train_arr = np.array(train_arr)

# 0-2: color item
# 3-8: deviation, uniformity, smoothness, entropy, skewness, kurtosis
#used_rows = [0, 1, 2, 3, 4, 6, 8]
used_rows = range(9)
train_arr = train_arr[:,used_rows,:]
print train_arr.shape

label_arr = []
for i, img in enumerate(os.listdir(os.path.join(train_path, 'label'))[:pic_num]):
	img_path = os.path.join(train_path, 'label', img)
	img = cv2.imread(img_path, 0)
	img = cv2.resize(img, (img_row, img_col))
	img_reshape = np.reshape(img, (img.shape[0]*img.shape[1], 1))
	img_squeeze = np.squeeze(img_reshape)
	label_arr.append(img_squeeze/255)
	print i, img_path
label_arr = np.array(label_arr)
print label_arr.shape

# Form the new matrix
new_x, new_y = train_arr[0,:,:], label_arr[0,:]
for i in range(train_arr.shape[0]):
	if i==0:
		new_x, new_y = deal_with_inbalance(train_arr[0,:,:], label_arr[0,:])
	else:
		tmp_x, tmp_y = deal_with_inbalance(train_arr[i,:,:], label_arr[i,:])
		new_x = np.concatenate((new_x, tmp_x), axis=1)
		new_y = np.append(new_y, tmp_y, axis=0)


new_x = np.squeeze(new_x).T
new_y = to_categorical(np.squeeze(new_y))

print new_x.shape, new_y.shape

model = build_model(input_dims=len(used_rows))

earlyStopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(new_x[:len(new_y)//10,:], new_y[:len(new_y)//10], batch_size=128, nb_epoch=100,verbose=1, validation_split=0.2, shuffle=True, callbacks=[earlyStopping])

img_test = gene_matrix('2.jpg', 128, 128, size=BATCH_SIZE, color=color)
print img_test.shape
img_ori = img_test[:3, :].T
print img_ori.shape
img_ori = np.reshape(img_ori, (128, 128, 3))
img_ori = img_ori*255
img_ori = img_ori.astype(int)
print img_ori.shape

with open('ori.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerows(img_ori)
cv2.imwrite('ori.jpg', img_ori)
img_test = img_test[used_rows]
print img_test.shape
pred = model.predict(img_test.T)
pred = np.argmax(pred, axis=1)
print pred.shape
img_test_ori = np.reshape(pred, (128, 128))
img_hsv = cv2.cvtColor(img_test_ori, cv2.COLOR_BGR2HSV)
img_test_ori = cv2.inRange(img_hsv, min_range, max_range)
cv2.imwrite('test.jpg', img_test_ori)


model_json = model.to_json()
with open("model_hybrid_skin.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_hybrid_skin.h5")
with open('train_history_hybrid_skin.json', 'w') as fp:
	json.dump(history.history, fp)
print("Saved model to disk")