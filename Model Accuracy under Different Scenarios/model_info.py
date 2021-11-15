from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import sys, logging
import time
from datetime import datetime
import pytz
import numpy as np

import sys
sys.path.append('..')
import parameters as param

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import tensorflow.keras as keras
from tensorflow.keras import backend as K

## load trained model
from tensorflow.keras.models import load_model

from helper import load_data

from tensorflow.python.client import device_lib


def accuracy(model, x, labels):
    assert (x.shape[0] == labels.shape[0])
    num = x.shape[0]
    y = model.predict(x)
    y = y.argmax(axis=-1)
    labels = labels.argmax(axis=-1)
    idx = (labels == y)
    return 100 * np.sum(idx) / num

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


## custom time zone for logger
def customTime(*args):
    utc_dt = pytz.utc.localize(datetime.utcnow())
    converted = utc_dt.astimezone(pytz.timezone("Singapore"))
    return converted.timetuple()

if __name__ == "__main__" : 

    os.makedirs("model_info/", exist_ok=True)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join("model_info/", 'output.log')),
            logging.StreamHandler()
        ])

    logging.Formatter.converter = customTime
    logger = logging.getLogger("model_information")
    
    logger.info("")
    logger.info("Use GPU: {}".format(len(get_available_gpus()) > 0))
    if len(get_available_gpus()) > 0 :
        logger.info("Available GPUs: {}".format(get_available_gpus()))


    datasets = ['mnist', 'cifar', 'svhn', 'eurosat']
    model_dict = {
                'mnist': ['lenet1', 'lenet4', 'lenet5'],
                'cifar': ['vgg16', 'resnet20', 'resnet56'],
                'svhn' : ['svhn_model', 'svhn_first', 'svhn_second'],
                'eurosat': ['resnet56']
                }

    batch_size = 256

    models = []
    layers = []
    parameters = []
    accuracies = []

    for dataset_name in datasets :
        for model_name in model_dict[dataset_name] :

            ## Load benign images
            x_train, y_train, x_test, y_test = load_data(dataset_name)

            ## Load keras pretrained model for the specific dataset
            model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
            model = load_model(model_path)

            # model.summary()
            
            logger.info("Dataset: {}".format(dataset_name))
            logger.info("Model: {}".format(model_name))
            # logger.info("Batch size: {}".format(batch_size))

            ## Check the accuracy of the original model on benign images
            acc = accuracy(model, x_test, y_test)
            logger.info("Model accuracy on benign images: {:.2f}%".format(acc))

            models.append(model_name)
            layers.append(len(model.layers))
            parameters.append(model.count_params())
            accuracies.append(acc)


    logger.info("{} & {} & {} &".format("model", "layer", "parameter", "acc"))
    for model, layer, parameter, acc in zip(models, layers, parameters, accuracies) :
        logger.info("& {} \t & {:,} \t & {:.2f}\% \t\\\\ {}".format(layer, parameter, acc, "\\hline" if model in ["lenet5", "resnet56", "svhn_second"] else ""))
        # logger.info("{} \t & {} \t & {:,} \t & {:.2f}\% \t\\\\ {}".format(model, layer, parameter, acc, "\\hline" if model in ["lenet5", "resnet56", "svhn_second"] else ""))

    