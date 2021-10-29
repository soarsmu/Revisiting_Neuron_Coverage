import argparse
import os
import random

import warnings
import sys

warnings.filterwarnings("ignore")

from keras import backend as K
import numpy as np
from PIL import Image, ImageFilter
from skimage.measure import compare_ssim as SSIM
import keras
from util import get_model
from helper import load_data, retrain

import tensorflow as tf
import os


DATA_DIR = "../data/"
MODEL_DIR = "../models/"




if __name__ == '__main__':

    dataset = 'mnist'
    model_name = 'lenet1'

    datasets = ['mnist', 'cifar', 'svhn']
    model_dict = {
                'mnist': ['lenet1', 'lenet4', 'lenet5'],
                'cifar': ['vgg16'], # , 'resnet20'
                'svhn' : ['svhn_model', 'svhn_second', 'svhn_first']
                }
    
    from keras.models import load_model
    for dataset in datasets:
        for model_name in model_dict[dataset]:
            x_train, y_train, x_test, y_test = load_data(dataset)
            # ## load mine trained model
            model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset, model_name)
            model = load_model(model_path)
            model.summary()
            folder_to_store = 'fuzzing/{}/{}/'.format(dataset, model_name)
            index = np.load(folder_to_store + 'nc_index_{}.npy'.format(4), allow_pickle=True).item()
            for y, x in index.items():
                x_train = np.concatenate((x_train, np.expand_dims(x, axis=0)), axis=0)
                y_train = np.concatenate((y_train, np.expand_dims(y_train[y], axis=0)), axis=0)

            retrained_model = retrain(model, x_train, y_train, x_test, y_test, batch_size=256, epochs=60)
            retrained_model.save('new_model/dp_{}.h5'.format(model_name))





