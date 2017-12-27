# ECE544 NA FA17 project Distracted Driver Action Recognition
# vgg 16 implementation
# author: Fangwen Wu

import tensorflow as tf
import numpy as np
from models.vgg16 import Vgg16
from train_eval_model import train_model, eval_model


def main(_):
    image_size = 224
    learning_rate = 0.0001
    batch_size = 32
    training_epoch = 10
    keep_prob = 0.8

    # Build model
    model = Vgg16(image_size, image_size)

    # Start training
    model = train_model(model,learning_rate, batch_size, training_epoch, keep_prob)

    # Start testing
    eval_model(model)


if __name__ == "__main__":
    tf.app.run()
