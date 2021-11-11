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


from parameters import classifier_params, attack_params
from helper import load_data

import art
from art.estimators.classification import TensorFlowV2Classifier

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

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def call_function_by_attack_name(attack_name):
    if attack_name not in param.ATTACK_NAMES:
        print('Unsupported attack: {}'.format(attack_name))
        sys.exit(1)
    return {
        param.APGD: AutoProjectedGradientDescent,
        param.BIM: BasicIterativeMethod,
        param.CW: CarliniLInfMethod,
        param.DF: DeepFool,
        param.FGSM: FastGradientMethod,
        param.JSMA: SaliencyMapMethod,
        param.NF: NewtonFool,
        param.PGD: ProjectedGradientDescent,
        param.SA: SquareAttack,
        param.SHA: ShadowAttack,
        param.ST: SpatialTransformation,
        param.WA: Wasserstein
    }[attack_name]

# integrate all attack method in one function and only construct graph once
def gen_adv_data(model, x, y, attack_name, dataset_name, batch_size=2048):
    logging.getLogger().setLevel(logging.CRITICAL)
    
    classifier_param = classifier_params[dataset_name]
    classifier = TensorFlowV2Classifier(model=model, **classifier_param)
    
    attack_param = attack_params[attack_name][dataset_name]
    if attack_name not in [param.ST] :
        if "batch_size" not in attack_param :
            attack_param["batch_size"] = batch_size
    if attack_name not in [param.FGSM, param.BIM] : ## some attacks don't have verbose parameter, e.g. bim
        attack_param["verbose"] = VERBOSE
    attack = call_function_by_attack_name(attack_name)(classifier, **attack_param)
    
    data_num = x.shape[0]
    adv_x = attack.generate(x=x, y=y)
    
    logging.getLogger().setLevel(logging.INFO)
    return adv_x


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
        '--dataset', help="Model Architecture", type=str, default="mnist")
    parser.add_argument(
        '--model', help="Model Architecture", type=str, default="lenet1")
    parser.add_argument(
        '--attack', help="Adversarial examples", type=str, default="fgsm")
    parser.add_argument(
        '--batch_size', help="batch size for generating adversarial examples", type=int, default=1024)

    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    attack_name = args.attack
    
    ## Prepare directory for saving adversarial images and logging
    adv_dir = "{}{}/adv/{}/{}/".format(
        param.DATA_DIR, dataset_name, model_name, attack_name)
    if not os.path.exists(adv_dir):
        os.makedirs(adv_dir)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(adv_dir, 'output.log')),
            logging.StreamHandler()
        ])

    logging.Formatter.converter = customTime
    logger = logging.getLogger("adversarial_images_generation")
    

    ## Load benign images
    x_train, y_train, x_test, y_test = load_data(dataset_name)

    ## Load keras pretrained model for the specific dataset
    model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
    model = load_model(model_path)
    # model.summary()
    
    logger.info("")
    logger.info("Generating Adversarial Images")
    logger.info("Use GPU: {}".format(len(get_available_gpus()) > 0))
    if len(get_available_gpus()) > 0 :
        logger.info("Available GPUs: {}".format(get_available_gpus()))

    logger.info("Dataset: {}".format(dataset_name))
    logger.info("Model: {}".format(model_name))
    logger.info("Attack: {}".format(attack_name))

    ## Check the accuracy of the original model on benign images
    acc = accuracy(model, x_test, y_test)
    logger.info("Model accuracy on benign images: {:.2f}%".format(acc))

    ## Generate adversarial images
    x_adv = gen_adv_data(model, x_test, y_test, attack_name,
                    dataset_name, args.batch_size)
    
    ## Check the accuracy of the original model on adversarial images
    acc = accuracy(model, x_adv, y_test)
    logger.info("Model accuracy on adversarial images: {:.2f}%".format(acc))
    
    ## Save the adversarial images into external file
    x_adv_path = "{}x_test.npy".format(adv_dir)
    np.save(x_adv_path, x_adv)

    ## Note: y_test will exactly be the same with the benign y_test
    ##       thus it's not a must to save the y_test
    y_adv_path = "{}y_test.npy".format(adv_dir)
    np.save(y_adv_path, y_test)

    logger.info("Adversarial images are saved at {}".format(adv_dir))
