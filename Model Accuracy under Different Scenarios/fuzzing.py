import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
import os

DATA_DIR = "../data/"
MODEL_DIR = "../models/"




import argparse
import os

import shutil
import warnings
from helper import Coverage
from helper import mutate, load_data, softmax, compare_nc, check_data_path

warnings.filterwarnings("ignore")



if __name__ == '__main__':

    datasets = ['mnist', 'cifar', 'svhn']
    model_dict = {
                'mnist': ['lenet1', 'lenet4', 'lenet5'],
                'cifar': ['vgg16', 'resnet20'], 
                'svhn' : ['svhn_model', 'svhn_second', 'svhn_first']
                }

    # Check path
    for dataset_name in model_dict.keys():
        # verify data path
        check_data_path(dataset_name)
        # verify model path
        for model_name in model_dict[dataset_name]:
            model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset_name, model_name)
            assert os.path.exists(model_path)

    from keras.models import load_model

    for dataset in datasets:
        for model_name in model_dict[dataset]:
            # load original model
            model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset, model_name)
            model = load_model(model_path)
            # get model layer
            model_layer = len(model.layers) - 1

            # load dataset
            x_train, y_train, x_test, y_test = load_data(dataset)


            for order_number in range(0, 10):
                nc_index = {}
                nc_number = 0
                for i in range(3000*order_number, 3000*(order_number+1)):
                    new_image = mutate(x_train[i], dataset)

                    if i == 5000*order_number+1000 or i == 5000*order_number+3000:
                        print("-------------------------------------THIS IS {}-------------------------------------".format(i))
                    if softmax(model.predict(np.expand_dims(new_image, axis=0))).argmax(axis=-1) != softmax(model.predict(np.expand_dims(x_train[i], axis=0))).argmax(axis=-1):

                        nc_symbol = compare_nc(model, x_train, y_train, x_test, y_test, new_image, x_train[i], model_layer)

                        if nc_symbol == True:
                            nc_index[i] = new_image
                            nc_number += 1

                print(nc_number)
                
                ### save data
                folder_to_store = 'fuzzing/{}/{}/'.format(dataset, model_name)
                os.makedirs(folder_to_store, exist_ok=True)
                np.save(folder_to_store +  '/nc_index_{}.npy'.format(order_number), nc_index)













