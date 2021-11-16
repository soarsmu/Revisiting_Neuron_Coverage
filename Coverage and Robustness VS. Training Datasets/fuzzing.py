import sys
sys.path.append('..')
import parameters as param

from keras.models import load_model
    
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
from helper import load_data, softmax, compare_nc, mutate, differentiable_mutate, nondifferentiable_mutate


warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='lenet1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--mutation", default='deephunter', type=str, choices=['deephunter', 'differentiable','nondifferentiable'])

    args = parser.parse_args()
    model_name = args.model
    
    dataset_name = args.dataset

    # load dataset
    x_train, y_train, x_test, y_test = load_data(dataset_name)

    # import model
    model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
    model = load_model(model_path)
    model_layer = len(model.layers) - 1

    folder_to_store = '{}{}/fuzzing_{}/{}/'.format(param.DATA_DIR, dataset_name, args.mutation, model_name)
    os.makedirs(folder_to_store, exist_ok=True)

    fuzzing_image_path = folder_to_store + "new_images.npy"

    if os.path.exists(fuzzing_image_path):
        new_images = np.load(fuzzing_image_path)
        print(f"Log: Load mutantions from {fuzzing_image_path}.")
    else:
        print("Log: Start do transformation in images")
        new_images = []
        
        for i in tqdm(range(len(x_train))):
            if args.mutation == "deephunter" :
                new_images.append(mutate(x_train[i]))
            elif args.mutation == "differentiable" : 
                new_images.append(differentiable_mutate(x_train[i]))
            elif args.mutation == "nondifferentiable" : 
                new_images.append(nondifferentiable_mutate(x_train[i]))

        np.save(fuzzing_image_path, new_images)
        print(f"Log: Save mutantions into {fuzzing_image_path}")

    print("Log: Find adversarial examples")

    for order_number in range(10):

        print("Log: order_number: {}".format(order_number))

        nc_index_path = f'{folder_to_store}/nc_index_{order_number}.npy'
        no_nc_index_path = f'{folder_to_store}/no_nc_index_{order_number}.npy'

        if os.path.exists(nc_index_path) and os.path.exists(no_nc_index_path) :
            print(f"Log: Images are already generated in {folder_to_store}/nc_index_{order_number}\n")
            print(f"Log: Images are already generated in {folder_to_store}/no_nc_index_{order_number} \n\n")
        else :
            nc_index = {}
            no_nc_index = {}
            nc_number = 0
            no_nc_number = 0

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
            print(f"Log: Save in {folder_to_store}/nc_index_{order_number}\n")
            np.save(nc_index_path, nc_index)
            
            print("Log: new image can NOT cover more neurons: {}".format(no_nc_number))
            print(f"Log: Save in {folder_to_store}/no_nc_index_{order_number} \n\n")
            np.save(no_nc_index_path, no_nc_index)

        
