import sys
sys.path.append('..')
import parameters as param

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import os
from helper import AttackEvaluate
from helper import load_data, retrain

####for solving some specific problems, don't care
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from keras.models import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='lenet1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--order_number", default=10, type=int)
    parser.add_argument("--mutation", default='deephunter', type=str, choices=['deephunter', 'differentiable','nondifferentiable'])

    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset
    order_number = args.order_number

    x_train, y_train, x_test, y_test = load_data(dataset_name)
    
    ## load trained model
    model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
    model = load_model(model_path)
    model_layer = len(model.layers)

    # model.summary()

    folder_to_store = '{}{}/fuzzing_{}/{}/'.format(param.DATA_DIR, dataset_name, args.mutation, model_name)

    for i in range(order_number):
        index = np.load(f'{folder_to_store}/nc_index_{i}.npy', allow_pickle=True).item()
        for y, x in index.items():
            x_train = np.concatenate((x_train, np.expand_dims(x, axis=0)), axis=0)
            y_train = np.concatenate((y_train, np.expand_dims(y_train[y], axis=0)), axis=0)

    retrained_model_path = f'{param.MODEL_DIR}{dataset_name}/adv_{model_name}_{args.mutation}.h5'
    
    retrained_model = retrain(model, x_train, y_train, x_test, y_test, batch_size=32, epochs=60)
    retrained_model.save(retrained_model_path)
