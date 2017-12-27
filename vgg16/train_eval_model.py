# ECE544 project
# fwu11
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import datetime
from utils import io_tools
from random import shuffle
import cv2

def train_model(model,learning_rate, batch_size,
                training_epoch, keep_prob):
    data_txt_file='data/driver_imgs_list.csv'
    image_data_path='data/train'
    rows = 224
    cols = 224

    saver = tf.train.Saver()
    # load saved model if any
    if os.path.isfile('models/model.ckpt'):
        load_path = saver.restore(model.session, 'models/model.ckpt')

    files = io_tools.load_train(
            data_txt_file, image_data_path)

    num_itr = (len(files)-1)//32 + 1
    for epoch in range(training_epoch):

        # read 32 images at a time
        epoch_loss = 0
        epoch_accuracy = 0
        shuffle(files)
        for i in range(0, len(files), 32):
            batch_x = []
            batch_y = []

            cur_batch_files = files[i: i+32]
            if len(cur_batch_files) != 32:
                continue
            for line in cur_batch_files:
                line = line.strip()
                path = image_data_path + '/' + str(line.split(',')[1]) + '/' + str(line.split(',')[2])
                batch_x.append(io_tools.read_image(path, rows, cols))
                batch_y.append(int(line.split(',')[1][1]))



            batch_x = np.array(batch_x, dtype=np.float32)
            batch_y = np.array(batch_y, dtype=np.float32)

            batch_y = tf.keras.utils.to_categorical(batch_y, 10)
			
            mean_pixel = [103.939, 116.779, 123.68]

            for c in range(3):
                batch_x[:, :, :, c] = batch_x[:, :, :, c] - mean_pixel[c]
 
            _, loss, accuracy_train, ts = model.session.run(
                [model.update_op_tensor, model.loss_tensor, model.accuracy_tensor, model.outputs_tensor],
                feed_dict={model.x_placeholder: batch_x, model.y_placeholder: batch_y,
                           model.learning_rate_placeholder: learning_rate, model.keep_prob_placeholder: keep_prob, model.phase_train: True}
            )
            epoch_loss += loss
            epoch_accuracy += accuracy_train
            print("[epoch %d/%d] [itr: %d/%d]  drodout: %f, accuracy = %f, loss = %f" %
                  (epoch, training_epoch, i, len(files), keep_prob, accuracy_train, loss))

        print("[epoch %d completed] drodout: %f, epoch_accuracy = %f, loss = %f" %
                  (epoch, keep_prob, epoch_accuracy/num_itr, epoch_loss/num_itr))

    # save model
    save_path = saver.save(model.session, 'models/model.ckpt')
    print("Model saved in file: %s" % save_path)

    return model


def eval_model(model, batch_size=32):
    files = io_tools.load_test()

    N = len(files)
    yfull_test = np.zeros((N, 10))
    test_id = []

    print(N)
    
    # load model

    saver = tf.train.Saver()
    if os.path.isfile('models/model.ckpt'):
        load_path = saver.restore(model.session, 'models/model.ckpt')

    for i in range(0, N, batch_size):
        cur_batch_files = files[i: i+32]
        batch_x = []
        for line in cur_batch_files:
            fbase = os.path.basename(line)
            batch_x.append(io_tools.read_image(line))
            test_id.append(fbase)
        
        batch_x = np.array(batch_x, dtype=np.float32)

        mean_pixel = [103.939, 116.779, 123.68]
        for c in range(3):
            batch_x[:, :, :, c] = batch_x[:, :, :, c] - mean_pixel[c]

        prediction = model.session.run(model.outputs_tensor, feed_dict={
            model.x_placeholder: batch_x, model.keep_prob_placeholder: 1, model.phase_train: False})
        yfull_test[i:i+32, :] = prediction

        print("[%d/%d completed]"%(i, N))
    print('================================')
    print(len(test_id))
    print(yfull_test[0].shape)
    create_submission(yfull_test, test_id)


def create_submission(predictions, test_id):
    result = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                'c4', 'c5', 'c6', 'c7',
                                                'c8', 'c9'])
    result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
    cols = result.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    result = result[cols]

    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result.to_csv(sub_file, index=False)