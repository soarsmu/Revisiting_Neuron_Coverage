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

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import tensorflow.keras as keras
from tensorflow.keras import backend as K
## load mine trained model
from tensorflow.keras.models import load_model

import art
from art.estimators.classification import TensorFlowV2Classifier
from art.defences.trainer import AdversarialTrainer
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import CarliniLInfMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import BasicIterativeMethod
from art.attacks.evasion import SaliencyMapMethod
from art.attacks.evasion import AutoProjectedGradientDescent
from art.attacks.evasion import DeepFool, NewtonFool
from art.attacks.evasion import SquareAttack, SpatialTransformation
from art.attacks.evasion import ShadowAttack, Wasserstein



import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


## custom time zone for logger
def customTime(*args):
    utc_dt = pytz.utc.localize(datetime.utcnow())
    converted = utc_dt.astimezone(pytz.timezone("Singapore"))
    return converted.timetuple()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## [original from FSE author] for solving some specific problems, don't care
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)


# VERBOSE = False
VERBOSE = True

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
APGD = "apgd"
DF = "deepfool"
NF = "newtonfool"
SA = "squareattack"
SHA = "shadowattack"
ST = "spatialtransformation"
WA = "wasserstein"
ATTACK_NAMES = [APGD, BIM, CW, DF, FGSM, JSMA, NF, PGD, SA, SHA, ST, WA]
## Note:  already tried APGD, but it doesn't work

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

## classifier paramaters
## depend to the dataset used
classifier_params = {}
for dataset_name in DATASET_NAMES :
    classifier_params[dataset_name] = {"loss_object": loss_object, "train_step": train_step}

classifier_params[MNIST]["nb_classes"] = 10
classifier_params[MNIST]["input_shape"] = (28, 28, 1)
classifier_params[MNIST]["clip_values"] = (-0.5, 0.5)

classifier_params[CIFAR]["nb_classes"] = 10
classifier_params[CIFAR]["input_shape"] = (32, 32, 3)

classifier_params[SVHN]["nb_classes"] = 10
classifier_params[SVHN]["input_shape"] = (32, 32, 3)
classifier_params[SVHN]["clip_values"] = (-0.5, 0.5)



## attack parameters for generating adversarial images
## Note: I use the same format with FSE paper, i.e. params[<attack>][<dataset>]
##       we may change the format into params[<dataset>][<attack>]
##       to track the consistent epsilon for each dataset
attack_params = {}

## TO DO: read the paper for each attack
##        make sure to use the correct parameters
##        for each combination of attack and dataset

# empty means using the original parameters for ART
attack_params[APGD] = {}
for dataset_name in DATASET_NAMES:
    attack_params[APGD][dataset_name] = {"loss_type": "cross_entropy"}

attack_params[CW] = {}
for dataset_name in DATASET_NAMES :
    attack_params[CW][dataset_name] = {}

attack_params[DF] = {}
for dataset_name in DATASET_NAMES:
    attack_params[DF][dataset_name] = {"batch_size": 256}

attack_params[NF] = {}
for dataset_name in DATASET_NAMES:
    attack_params[NF][dataset_name] = {"batch_size": 256}

attack_params[JSMA] = {}
for dataset_name in DATASET_NAMES:
    attack_params[JSMA][dataset_name] = {}

attack_params[SA] = {}
for dataset_name in DATASET_NAMES:
    attack_params[SA][dataset_name] = {}

attack_params[SHA] = {}
for dataset_name in DATASET_NAMES:
    attack_params[SHA][dataset_name] = {"batch_size": 1}

attack_params[ST] = {}
for dataset_name in DATASET_NAMES:
    attack_params[ST][dataset_name] = {}

attack_params[WA] = {}
for dataset_name in DATASET_NAMES:
    attack_params[WA][dataset_name] = {}


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
        APGD: AutoProjectedGradientDescent,
        BIM: BasicIterativeMethod,
        CW: CarliniLInfMethod,
        DF: DeepFool,
        FGSM: FastGradientMethod,
        JSMA: SaliencyMapMethod,
        NF: NewtonFool,
        PGD: ProjectedGradientDescent,
        SA: SquareAttack,
        SHA: ShadowAttack,
        ST: SpatialTransformation,
        WA: Wasserstein
    }[attack_name]


# integrate all attack method in one function and only construct graph once
def gen_adv_data(model, x, y, attack_name, dataset_name, batch_size=2048):
    
    classifier_param = classifier_params[dataset_name]
    classifier = TensorFlowV2Classifier(model=model, **classifier_param)
    
    attack_param = attack_params[attack_name][dataset_name]
    if attack_name not in [ST] :
        if "batch_size" not in attack_param :
            attack_param["batch_size"] = batch_size
    if attack_name not in [FGSM, BIM] : ## some attacks don't have verbose parameter, e.g. bim
        attack_param["verbose"] = VERBOSE

    attack = call_function_by_attack_name(attack_name)(classifier, **attack_param)
    
    data_num = x.shape[0]
    adv_x = attack.generate(x=x, y=y)
    
    logging.getLogger().setLevel(logging.INFO)
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


def adv_retrain(attack_name, dataset, model_name, batch_size=512):
    '''adversrial retrain model'''
    x_train, y_train, x_test, y_test = load_data(dataset)

    ## Load keras pretrained model for the specific dataset
    model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset, model_name)
    model = load_model(model_path)
    model.summary()

    # model.compile(
    #     loss='categorical_crossentropy',
    #     optimizer='adam',
    #     metrics=['accuracy']
    # )

    labels_true = np.argmax(y_test, axis=1)
    labels_test = np.argmax(model.predict(x_test), axis=1)
    print('Accuracy test set: %.2f%%' % (np.sum(labels_test == labels_true) / x_test.shape[0] * 100))

    classifier_param = classifier_params[dataset_name]
    classifier = TensorFlowV2Classifier(model=model, **classifier_param)

    attack_param = attack_params[attack_name][dataset_name]
    if attack_name not in [ST] :
        if "batch_size" not in attack_param :
            attack_param["batch_size"] = batch_size
    if attack_name not in [FGSM, BIM] : ## some attacks don't have verbose parameter, e.g. bim
        attack_param["verbose"] = VERBOSE
    
    attack = AutoProjectedGradientDescent(classifier, **attack_param)
    # attack = call_function_by_attack_name(attack_name)(classifier, **attack_param)

    x_test_pgd = attack.generate(x_test, y_test)


    # Evaluate the benign trained model on adv test set
    labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
    print('Accuracy on original ' + attack_name + ' adversarial samples: %.2f%%' %
        (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100))

    # Adversarial Training
    trainer = AdversarialTrainer(classifier, attack, ratio=1.0)
    if dataset_name == 'svhn':
        nb_epochs = 20
    else:
        nb_epochs = 80
    trainer.fit(x_train, y_train, nb_epochs=nb_epochs, batch_size=512)

    # Save model
    classifier.save(filename= 'adv_' + model_name + '_' + attack_name + '.h5', path="{}{}".format(MODEL_DIR, dataset))


    # Evaluate the adversarially trained model on clean test set
    labels_true = np.argmax(y_test, axis=1)
    labels_test = np.argmax(classifier.predict(x_test), axis=1)
    print('Accuracy test set: %.2f%%' % (np.sum(labels_test == labels_true) / x_test.shape[0] * 100))

    # Evaluate the adversarially trained model on original adversarial samples
    labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
    print('Accuracy on original ' + attack_name + ' adversarial samples: %.2f%%' %
        (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100))

    # Evaluate the adversarially trained model on fresh adversarial samples produced on the adversarially trained model
    x_test_pgd = attack.generate(x_test, y_test)
    labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
    print('Accuracy on new ' + attack_name + ' adversarial samples: %.2f%%' % (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarially retrain model')
    parser.add_argument('-dataset', help="dataset to use", choices=['mnist', 'cifar', 'svhn'])
    parser.add_argument('-model', help="target model to attack",
                        choices=['vgg16', 'resnet20', 'lenet1', 'lenet4', 'lenet5', 'svhn_model',
                                 'svhn_first', 'svhn_second'])
    parser.add_argument('-attack', help="dataset to use")
    args = parser.parse_args()

    attack_name = args.attack
    dataset_name = args.dataset
    dataset = dataset_name
    model_name = args.model


    batch_size = 1024


    x_train, y_train, x_test, y_test = load_data(dataset)

    ## Load keras pretrained model for the specific dataset
    model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset, model_name)
    model = load_model(model_path)
    model.summary()


    labels_true = np.argmax(y_test, axis=1)
    labels_test = np.argmax(model.predict(x_test), axis=1)
    print('Accuracy test set: %.2f%%' % (np.sum(labels_test == labels_true) / x_test.shape[0] * 100))

    classifier_param = classifier_params[dataset_name]
    classifier = TensorFlowV2Classifier(model=model, **classifier_param)

    attack_param = attack_params[attack_name][dataset_name]
    if attack_name not in [ST] :
        if "batch_size" not in attack_param :
            attack_param["batch_size"] = batch_size
    if attack_name not in [FGSM, BIM] : ## some attacks don't have verbose parameter, e.g. bim
        attack_param["verbose"] = VERBOSE
    
    attack = call_function_by_attack_name(attack_name)(classifier, **attack_param)

    # Adversarial Training
    trainer = AdversarialTrainer(classifier, attack, ratio=1.0)
    if dataset_name == 'svhn':
        nb_epochs = 20
    else:
        nb_epochs = 80
    trainer.fit(x_train, y_train, nb_epochs=nb_epochs, batch_size=512)







