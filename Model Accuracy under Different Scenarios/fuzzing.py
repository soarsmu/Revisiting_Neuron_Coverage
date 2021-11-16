from helper import mutate, load_data, softmax, compare_nc, check_data_path
from keras.models import load_model
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
import os
import argparse
import os, sys

import shutil
import warnings
from helper import Coverage

warnings.filterwarnings("ignore")

sys.path.append('..')

import parameters as param


def run_fuzzing(dataset_name, model, x_train, y_train, x_test, y_test, model_layer, folder_to_store, order_numbers=10):
    
    for order_number in range(0, order_numbers):
        
        file_path = '{}nc_index_{}.npy'.format(folder_to_store, order_number)
        
        # only perform fuzzing if the file does not exist
        if not os.path.exists(file_path) :
            nc_index = {}
            nc_number = 0
            lower_bound = 3000 * order_number
            upper_bound = 3000 * (order_number+1)

            if lower_bound > len(x_train): lower_bound = len(x_train)

            if upper_bound > len(x_train): upper_bound = len(x_train)

            for i in range(lower_bound, upper_bound):
                new_image = mutate(x_train[i], dataset_name)

                if i == 5000*order_number+1000 or i == 5000*order_number+3000:
                    print("-------------------------------------THIS IS {}-------------------------------------".format(i))
                if softmax(model.predict(np.expand_dims(new_image, axis=0))).argmax(axis=-1) != softmax(model.predict(np.expand_dims(x_train[i], axis=0))).argmax(axis=-1):

                    nc_symbol = compare_nc(
                        model, x_train, y_train, x_test, y_test, new_image, x_train[i], model_layer)

                    if nc_symbol == True:
                        nc_index[i] = new_image
                        nc_number += 1

            print(nc_number)
            np.save(file_path, nc_index)


if __name__ == '__main__':

    # datasets = ['mnist', 'cifar', 'svhn']
    # model_dict = {
    #             'mnist': ['lenet1', 'lenet4', 'lenet5'],
    #             'cifar': ['vgg16', 'resnet20'], 
    #             'svhn' : ['svhn_model', 'svhn_second', 'svhn_first']
    #             }

    datasets = ['eurosat']
    model_dict = {
        'eurosat': ['resnet56']
    }

    # Check path
    for dataset_name in model_dict.keys():
        # verify data path
        check_data_path(dataset_name)
        # verify model path
        for model_name in model_dict[dataset_name]:
            model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
            assert os.path.exists(model_path)

    
    for dataset_name in datasets:
        for model_name in model_dict[dataset_name]:
            # load original model
            model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
            model = load_model(model_path)
            # get model layer
            model_layer = len(model.layers) - 1

            # load dataset
            x_train, y_train, x_test, y_test = load_data(dataset_name)

            ### save data
            folder_to_store = '{}{}/fuzzing/{}/'.format(param.DATA_DIR, dataset_name, model_name)
            os.makedirs(folder_to_store, exist_ok=True)

            run_fuzzing(dataset_name, model, x_train, y_train, x_test,y_test, model_layer, folder_to_store)














