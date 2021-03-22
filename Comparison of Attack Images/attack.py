from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
from util import get_model
import time
import numpy as np
from keras import backend as K
import keras
## load mine trained model
from keras.models import load_model

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import ProjectedGradientDescent


import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DATA_DIR = "../data/"
MODEL_DIR = "../models/"

## [original from FSE author] for solving some specific problems, don't care
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

## attack parameters for generating adversarial images
attack_params = {}
attack_params['CW'] = {}
for dataset_name in ['mnist', 'cifar', 'svhn'] :
    attack_params['CW'][dataset_name] = {'binary_search_steps': 1,
                                         'learning_rate': .1,
                                         'max_iterations': 50,
                                         'initial_const': 10
                                         # 'clip_min': -0.5,
                                         # 'clip_max': 0.5
                                         }

attack_params['PGD'] = {}
attack_params['PGD']['mnist'] = {'eps': .3,
                      'eps_iter': .03,
                      'nb_iter': 20.,
                      'clip_min': -0.5,
                      'clip_max': 0.5
                      }
attack_params['PGD']['cifar'] = {'eps': 16. / 255.,
                      'eps_iter': 2. / 255.,
                      'nb_iter': 30.
                      # 'clip_min': -0.5,
                      # 'clip_max': 0.5
                      }
attack_params['PGD']['svhn'] = {'eps': 8. / 255.,
                      'eps_iter': 0.01,
                      'nb_iter': 30.,
                      'clip_min': -0.5,
                      'clip_max': 0.5
                      }


# def FGSM(model, x, y):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     fgsm = FastGradientMethod(model_wrap, sess=sess)
#     fgsm_params={'y':y, 'eps':0.2, 'clip_min':0., 'clip_max': 1.}
#     adv = fgsm.generate_np(x, **fgsm_params)
#     return adv
attack_params['FGSM'] = {}
attack_params['FGSM']['mnist'] = {'eps': .3,
                                 'clip_min': -0.5,
                                 'clip_max': 0.5
                                 }
attack_params['FGSM']['cifar'] = {'eps': 16. / 255.,
                                 # 'clip_min': -0.5,
                                 # 'clip_max': 0.5
                                 }
attack_params['FGSM']['svhn'] = {'eps': 8. / 255.,
                                'clip_min': -0.5,
                                'clip_max': 0.5
                                }


# def BIM(model, x, y):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     bim = BasicIterativeMethod(model_wrap, sess=sess)
#     bim_params={'eps_iter': 0.03, 'nb_iter': 10, 'y':y, 'clip_min': 0., 'clip_max': 1.}
#     adv = bim.generate_np(x, **bim_params)
#     return adv
attack_params['BIM'] = {}
for dataset_name in ['mnist', 'cifar', 'svhn']:
    attack_params['BIM'][dataset_name] = {
        'eps_iter': 0.03, 'nb_iter': 10, 'clip_min': 0., 'clip_max': 1.}


# def JSMA(model, x, y):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     jsma = SaliencyMapMethod(model_wrap, sess=sess)
#     jsma_params = {'theta':1., 'gamma': 0.1, 'clip_min':0., 'clip_max':1.}
#     adv = jsma.generate_np(x, **jsma_params)
#     return adv




# integrate all attack method in one function and only construct graph once
def gen_adv_data(model, x, y, attack_name, dataset_name, batch_size=2048):
    sess = K.get_session()
    model_wrap = KerasModelWrapper(model)
    params = attack_params[attack_name][dataset_name]
    if attack_name == 'CW':
        attack = CarliniWagnerL2(model_wrap, sess=sess)
    elif attack_name == 'PGD':
        attack = ProjectedGradientDescent(model_wrap, sess=sess)
    elif attack_name == 'FGSM':
        attack = FastGradientMethod(model_wrap, sess=sess)
    elif attack_name == 'BIM':
        attack = BasicIterativeMethod(model_wrap, sess=sess)
    else:
        print('Unsupported attack')
        sys.exit(1)
    
    data_num = x.shape[0]
    begin, end = 0, batch_size
    adv_x_all = np.zeros_like(x)
    # every time process batch_size
    while end < data_num:
        start_time = time.time()
        if attack_name in ["CW"] :
            adv_x = attack.generate_np(
                x_val=x[begin:end], y=y[begin:end], batch_size=batch_size, **params)
        elif attack_name in ["PGD", "FGSM", "BIM"]:
            adv_x = attack.generate_np(
                x_val=x[begin:end], y=y[begin:end], **params)
        adv_x_all[begin: end] = adv_x
        print(begin, end, "done")
        begin += batch_size
        end += batch_size
        end_time = time.time()
        print("time: ", end_time - start_time)

    # process the remaining
    if begin < data_num:
        start_time = time.time()
        if attack_name in ["CW"]:
            curr_batch_size = data_num - begin
            adv_x = attack.generate_np(
                x_val=x[begin:], y=y[begin:], batch_size=curr_batch_size, **params)
        elif attack_name in ["PGD", "FGSM", "BIM"]:
            adv_x = attack.generate_np(
                x_val=x[begin:], y=y[begin:], **params)
        adv_x_all[begin:] = adv_x
        print(begin, data_num, "done")
        end_time = time.time()
        print("time: ", end_time - start_time)

    return adv_x_all



# the data is in range(-.5, .5)
def load_data(dataset_name):
    assert (dataset_name.upper() in ['MNIST', 'CIFAR', 'SVHN'])
    dataset_name = dataset_name.lower()
    x_train = np.load(DATA_DIR + dataset_name + '/benign/x_train.npy')
    y_train = np.load(DATA_DIR + dataset_name + '/benign/y_train.npy')
    x_test = np.load(DATA_DIR + dataset_name + '/benign/x_test.npy')
    y_test = np.load(DATA_DIR + dataset_name + '/benign/y_test.npy')
    return x_train, y_train, x_test, y_test


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def accuracy(model, x, labels):
    assert (x.shape[0] == labels.shape[0])
    num = x.shape[0]
    y = model.predict(x)
    y = y.argmax(axis=-1)
    labels = labels.argmax(axis=-1)
    idx = (labels == y)
    print(np.sum(idx) / num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attack for DNN')
    parser.add_argument('-dataset', help="dataset to use",
                        choices=['mnist', 'cifar', 'svhn'])
    parser.add_argument('-model', help="target model to attack", choices=['vgg16', 'resnet20', 'lenet1', 'lenet4', 'lenet5', 'adv_lenet1', 'adv_lenet4',
                                                                          'adv_lenet5', 'adv_vgg16', 'adv_resnet20', 'svhn_model', 'adv_svhn_model', 'svhn_first', 'adv_svhn_first', 'svhn_second', 'adv_svhn_second'])
    parser.add_argument('-attack', help="attack model", choices=['CW', 'PGD'])
    parser.add_argument(
        '-batch_size', help="attack batch size", type=int, default=32)

    args = parser.parse_args()
    # args.dataset = 'cifar'
    # args.attack = 'PGD'

    dataset_name = "mnist"
    model_name = "lenet1"
    attack_name = "PGD"

    datasets = ['mnist', 'cifar', 'svhn']
    model_dict = {
        'mnist': ['lenet1', 'lenet4', 'lenet5'],
        'cifar': ['vgg16', 'resnet20'],
        'svhn': ['svhn_model', 'svhn_first', 'svhn_second']
    }
    
    # model_dict = {
    #     'mnist': ['lenet1']
    # }

    # model_dict = {
    #     'cifar': ['vgg16'],
    # }

    # attack_names = ["PGD", "CW", "FGSM", "BIM"]
    attack_names = ["PGD", "CW", "FGSM"]
    # attack_names = ['BIM']

    for dataset_name in datasets:
        if dataset_name in model_dict:
            for model_name in model_dict[dataset_name]:
                for attack_name in attack_names :
                    
                    ## Load benign images from mnist, cifar, or svhn
                    x_train, y_train, x_test, y_test = load_data(dataset_name)

                    ## Load keras pretrained model for the specific dataset
                    model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset_name, model_name)
                    model = load_model(model_path)
                    model.summary()

                    ## Check the accuracy of the original model on benign images
                    accuracy(model, x_test, y_test)

                    ## Generate adversarial images
                    adv = gen_adv_data(model, x_test, y_test, attack_name,
                                    dataset_name, args.batch_size)
                    
                    # accuracy(model, adv, y_test)

                    ## Save the adversarial images into external file
                    adv_dir = "{}{}/adv/{}/".format(DATA_DIR, dataset_name, model_name)
                    if not os.path.exists(adv_dir):
                        os.makedirs(adv_dir)
                    adv_path = "{}{}.npy".format(adv_dir, attack_name)
                    np.save(adv_path, adv)
