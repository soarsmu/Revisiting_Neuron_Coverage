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
from keras.models import load_model


sys.path.append('..')

import parameters as param

from fuzzing import run_fuzzing

if __name__ == '__main__':

    datasets = ['mnist', 'cifar', 'svhn']
    model_dict = {
                'mnist': ['lenet1', 'lenet4', 'lenet5'],
                'cifar': ['vgg16', 'resnet20'],
                'svhn' : ['svhn_model', 'svhn_second', 'svhn_first']
                }
    
    # datasets = ['cifar', 'eurosat']
    # model_dict = {
    #     'cifar': ['resnet56'],
    #     'eurosat': ['resnet20', 'resnet56'],
    # }
    
    for dataset_name in datasets:
        for model_name in model_dict[dataset_name]:

            print("Dataset: ", dataset_name)
            print("Model: ", model_name)

            x_train, y_train, x_test, y_test = load_data(dataset_name)
            
            # load trained model
            model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
            model = load_model(model_path)
            # model.summary()
            
            # get model layer
            model_layer = len(model.layers) - 1

            # prepare folder to store images generated from fuzzing
            folder_to_store = '{}{}/fuzzing/{}/'.format(param.DATA_DIR, dataset_name, model_name)
            os.makedirs(folder_to_store, exist_ok=True)

            # run fuzzing 
            run_fuzzing(dataset_name, model, x_train, y_train, x_test,
                        y_test, model_layer, folder_to_store)
            
            # load images generated from fuzzing
            index = np.load(folder_to_store + 'nc_index_{}.npy'.format(4), allow_pickle=True).item() 
            # To-Do: train on all images, instead of just one.

            
            for y, x in index.items():
                x_train = np.concatenate((x_train, np.expand_dims(x, axis=0)), axis=0)
                y_train = np.concatenate((y_train, np.expand_dims(y_train[y], axis=0)), axis=0)

            retrained_model_path = '{}{}/adv_{}_deephunter.h5'.format(
                param.MODEL_DIR, dataset_name, model_name)
            
            # only retrain if the retrained model is not exist
            if not os.path.exists(retrained_model_path) :

                # retrain model using augmented data from fuzzing
                retrained_model = retrain(model, x_train, y_train, x_test, y_test, batch_size=256, epochs=60)
                
                # save model for future usage
                retrained_model.save(retrained_model_path)





