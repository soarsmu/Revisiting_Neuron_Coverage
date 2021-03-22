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

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import CarliniLInfMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import BasicIterativeMethod
from art.attacks.evasion import SaliencyMapMethod


import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VERBOSE = False

DATA_DIR = "../data/"
MODEL_DIR = "../models/"

MNIST = "mnist"
CIFAR = "cifar"
SVHN = "svhn"

DATASET_NAMES = [MNIST, CIFAR, SVHN]

BIM = "bim"
CW = "cw"
FGSM = "fgsm"
JSMA = "jsma"
PGD = "pgd"
ATTACK_NAMES = [BIM, CW, FGSM, JSMA, PGD]

## [original from FSE author] for solving some specific problems, don't care
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


## classifier paramaters
## depend to the dataset used
classifier_params = {}
for dataset_name in DATASET_NAMES :
    classifier_params[dataset_name] = {}
classifier_params[MNIST] = {"clip_values": (-0.5, 0.5)}
classifier_params[SVHN] = {"clip_values": (-0.5, 0.5)}

## attack parameters for generating adversarial images
attack_params = {}


attack_params[CW] = {}
for dataset_name in DATASET_NAMES :
    attack_params[CW][dataset_name] = {}

attack_params[JSMA] = {}
for dataset_name in DATASET_NAMES:
    attack_params[JSMA][dataset_name] = {}

attack_params[PGD] = {}
attack_params[PGD][MNIST] = {'eps': .3,
                      'eps_step': .03,
                      'max_iter': 20
                      }
attack_params[PGD][CIFAR] = {'eps': 16. / 255.,
                      'eps_step': 2. / 255.,
                      'max_iter': 30
                      }
attack_params[PGD][SVHN] = {'eps': 8. / 255.,
                      'eps_step': 0.01,
                      'max_iter': 30
                      }

# use the same epsilon used in pgd
attack_params[BIM] = {}
attack_params[BIM][MNIST] = {'eps': .3
                                  }
attack_params[BIM][CIFAR] = {'eps': 16. / 255.
                                  }
attack_params[BIM][SVHN] = {'eps': 8. / 255.
                                 }


# use the same epsilon used in pgd
attack_params[FGSM] = {}
attack_params[FGSM][MNIST] = {'eps': .3
                                 }
attack_params[FGSM][CIFAR] = {'eps': 16. / 255.
                                 }
attack_params[FGSM][SVHN] = {'eps': 8. / 255.
                                }



def call_function_by_attack_name(attack_name):
    if attack_name not in ATTACK_NAMES:
        print('Unsupported attack: {}'.format(attack_name))
        sys.exit(1)
    return {
        FGSM: FastGradientMethod,
        PGD: ProjectedGradientDescent,
        BIM: BasicIterativeMethod,
        CW: CarliniLInfMethod,
        JSMA: SaliencyMapMethod
    }[attack_name]


# integrate all attack method in one function and only construct graph once
def gen_adv_data(model, x, y, attack_name, dataset_name, batch_size=2048):
    classifier_param = classifier_params[dataset_name]
    classifier = KerasClassifier(model, **classifier_param)
    
    attack_param = attack_params[attack_name][dataset_name]
    attack_param["batch_size"] = batch_size
    if attack_name in [CW, PGD] : ## some attacks don't have verbose parameter, e.g. bim
        attack_param["verbose"] = VERBOSE
    attack = call_function_by_attack_name(attack_name)(classifier, **attack_param)
    
    data_num = x.shape[0]
    adv_x = attack.generate(x=x, y=y)

    return adv_x



# the data is in range(-.5, .5)
def load_data(dataset_name):
    assert dataset_name in DATASET_NAMES
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
    return 100 * np.sum(idx) / num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack for DNN')
    parser.add_argument(
        '--batch_size', help="batch size for generating adversarial examples", type=int, default=1024)

    args = parser.parse_args()
    
    dataset_name = MNIST
    model_name = "lenet1"
    attack_name = PGD

    datasets = [MNIST, CIFAR, SVHN]

    # model_dict = {
    #     MNIST: ['lenet1', 'lenet4', 'lenet5'],
    #     CIFAR: ['vgg16', 'resnet20'],
    #     SVHN: ['svhn_model', 'svhn_first', 'svhn_second']
    # }
    
    model_dict = {
        MNIST: ['lenet1']
    }

    # model_dict = {
    #     CIFAR: ['vgg16'],
    # }

    attack_names = [PGD, CW, FGSM, BIM, JSMA]
    # attack_names = [PGD, CW, FGSM]
    # attack_names = [FGSM]
    # attack_names = [PGD]
    # attack_names = [BIM]
    # attack_names = [CW]
    # attack_names = [JSMA]

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

                    print("Dataset: {}".format(dataset_name))
                    print("Model: {}".format(model_name))
                    print("Attack: {}".format(attack_name))

                    ## Check the accuracy of the original model on benign images
                    acc = accuracy(model, x_test, y_test)
                    print("Model accuracy on benign images: {:.2f}%".format(acc))

                    ## Generate adversarial images
                    x_adv = gen_adv_data(model, x_test, y_test, attack_name,
                                    dataset_name, args.batch_size)
                    
                    ## Check the accuracy of the original model on adversarial images
                    acc = accuracy(model, x_adv, y_test)
                    print("Model accuracy on adversarial images: {:.2f}%".format(acc))
                    
                    ## Save the adversarial images into external file
                    adv_dir = "{}{}/adv/{}/".format(DATA_DIR, dataset_name, model_name)
                    if not os.path.exists(adv_dir):
                        os.makedirs(adv_dir)
                    adv_path = "{}{}.npy".format(adv_dir, attack_name)
                    np.save(adv_path, x_adv)
