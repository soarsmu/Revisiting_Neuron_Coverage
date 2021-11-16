import argparse
import os
import random
import copy

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

from simple_attack import simple_mutate

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple Retrain for DNN')
    parser.add_argument('--dataset', help="dataset used for generating adv examples", type=str, default="mnist")
    parser.add_argument('--model', help="model architecture", type=str, default="lenet1")
    parser.add_argument('--batch_size', help="batch_size", type=int, default=256)
    
    args = parser.parse_args()
                
    dataset_name = args.dataset
    model_name = args.model
    batch_size = args.batch_size

    np.random.seed(1)


    print("Dataset: ", dataset_name)
    print("Model: ", model_name)
    print("Batch size: ", batch_size)

    x_train, y_train, x_test, y_test = load_data(dataset_name)
    
    # load trained model
    model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
    model = load_model(model_path)
    # model.summary()
    
    # get model layer
    model_layer = len(model.layers) - 1

    x_adv = copy.deepcopy(x_train)
    for i in range(len(x_train)):
        new_image = simple_mutate(x_train[i], dataset_name)

        if x_adv.size == 0:
            x_adv = np.expand_dims(new_image, axis=0)
        else:
            x_adv = np.concatenate((x_adv, np.expand_dims(new_image, axis=0)), axis=0)

    x_adv = np.random.shuffle(x_adv)

    retrained_model_path = '{}{}/adv_{}_simple.h5'.format(
        param.MODEL_DIR, dataset_name, model_name)
    
    # only retrain if the retrained model is not exist
    if not os.path.exists(retrained_model_path) :

        # retrain model using augmented data from fuzzing
        retrained_model = retrain(model, x_train, y_train, x_test, y_test, batch_size=batch_size, epochs=60)
        
        # save model for future usage
        retrained_model.save(retrained_model_path)





