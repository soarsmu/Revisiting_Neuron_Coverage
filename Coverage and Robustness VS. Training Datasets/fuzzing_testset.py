import sys
sys.path.append('..')
import parameters as param

import numpy as np
import tensorflow as tf
import os
import argparse

from tqdm import tqdm
from keras.models import load_model
    
####for solving some specific problems, don't care
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import warnings
from helper import load_data, softmax, compare_nc, mutate, differentiable_mutate, nondifferentiable_mutate

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--model", default='lenet1', type=str)
    parser.add_argument("--mutation", default='deephunter', type=str, choices=['deephunter', 'differentiable','nondifferentiable'])

    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset

    print("Fuzzing on test data ....")
    print("Dataset: ", dataset_name)
    print("Model: ", model_name)
    print("Mutation: ", args.mutation)

    # load dataset
    x_train, y_train, x_test, y_test = load_data(dataset_name)

    # import model
    model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
    model = load_model(model_path)
    model_layer = len(model.layers) - 1

    store_path = "{}{}/adv/{}/fuzzing_{}/".format(param.DATA_DIR, dataset_name, model_name, args.mutation)
    
    os.makedirs(store_path, exist_ok=True)
    new_images = []
    for i in tqdm(range(len(x_test)), desc="transformation ......"):
        if args.mutation == "deephunter" :
            new_images.append(mutate(x_train[i]))
        elif args.mutation == "differentiable" : 
            new_images.append(differentiable_mutate(x_train[i]))
        elif args.mutation == "nondifferentiable" : 
            new_images.append(nondifferentiable_mutate(x_train[i]))


    for order_number in range(2):

        nc_index_path = os.path.join(store_path, 'nc_index_test_{}.npy'.format(order_number))

        if not os.path.exists(nc_index_path) :
            nc_index = {}
            nc_number = 0

            lower_bound = 5000 * order_number
            upper_bound = 5000 * (order_number+1)

            if lower_bound > len(new_images): lower_bound = len(new_images)

            if upper_bound > len(new_images): upper_bound = len(new_images)

            step = int((upper_bound - lower_bound) / 10)
            for i in tqdm(range(lower_bound, upper_bound, step), desc="Total progress:"):
              
                left_idx = i
                right_idx = min(i + step, upper_bound)

                for index, (pred_new, pred_old) in enumerate(zip(softmax(model.predict(np.array(new_images[left_idx:right_idx]))).argmax(axis=-1), softmax(model.predict(x_test[left_idx:right_idx])).argmax(axis=-1))):
                    nc_symbol = compare_nc(model, x_train, y_train, x_test, y_test, new_images[i+index], x_test[i+index], model_layer)
                    if nc_symbol == True:
                        nc_index[i+index] = new_images[i+index]
                        nc_number += 1

            print("Log: new image can cover more neurons: {}".format(nc_number))
            np.save(nc_index_path, nc_index)

    for order_number in range(2):
        index = np.load(os.path.join(store_path, 'nc_index_test_{}.npy'.format(order_number)), allow_pickle=True).item()
        for y, x in index.items():
            x_test[y] = x

    np.save(os.path.join(store_path, 'x_test.npy'), x_test)













