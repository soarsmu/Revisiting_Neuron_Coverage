import os
import numpy as np
import tensorflow as tf
import argparse
import time

from tqdm import tqdm

####for solving some specific problems, don't care
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import warnings
from helper import load_data, softmax, compare_nc, mutate, get_all_layers_outputs

import multiprocessing
from multiprocessing import Pool
manager = multiprocessing.Manager()
queue = manager.Queue()

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='lenet1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)

    args = parser.parse_args()
    model_name = args.model_name
    
    dataset_name = args.dataset

    # load dataset
    x_train, y_train, x_test, y_test = load_data(dataset_name)

    # import model
    from keras.models import load_model
    MODEL_DIR = "../models/"
    model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset_name, model_name)
    model = load_model(model_path)
    model_layer = len(model.layers)

    if os.path.exists("fuzzing/new_images.npy"):
        new_images = np.load("fuzzing/new_images.npy")
        print("Log: Load mutantions from fuzzing/new_images.npy.")
    else:
        print("Log: Start do transformation in images")
        new_images = []
        
        for i in tqdm(range(len(x_train))):
            new_images.append(mutate(x_train[i]))
            
        np.save("fuzzing/new_images.npy", new_images)
        print("Log: Save mutantions into fuzzing/new_images.npy")

    print("Log: Find adversarial examples")

    for order_number in range(10):
        nc_index = {}
        no_nc_index = {}
        nc_number = 0
        no_nc_number = 0

        print("Log: order_number: {}".format(order_number))
        
        for i in tqdm(range(5000*order_number, 5000*(order_number+1), 500), desc="Total progress:"):
            for index, (pred_new, pred_old) in enumerate(zip(softmax(model.predict(np.array(new_images[i:i+500]))).argmax(axis=-1), softmax(model.predict(x_train[i:i+500])).argmax(axis=-1))):
                
                if pred_new != pred_old:
                    nc_symbol = compare_nc(model, x_train, y_train, x_test, y_test, new_images[i+index], x_train[i+index], model_layer)
                    if nc_symbol == True:
                        # new image can cover more neurons
                        nc_index[i+index] = new_images[i+index]
                        nc_number += 1
                    else:
                        no_nc_index[i+index] = new_images[i+index]
                        no_nc_number += 1

        print("Log: new image can cover more neurons: {}".format(nc_number))
        print("Log: new image can NOT cover more neurons: {}".format(no_nc_number))
        print("Log: Save in fuzzing/nc_index_{}, fuzzing/nc_index_{} \n\n".format(order_number, order_number))
        np.save('fuzzing/nc_index_{}.npy'.format(order_number), nc_index)
        np.save('fuzzing/no_nc_index_{}.npy'.format(order_number), no_nc_index)

        
