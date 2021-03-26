import numpy as np
import parameters as param

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


DATA_DIR = param.DATA_DIR
DATASET_NAMES = param.DATASET_NAMES

def load_data(dataset_name):
    assert dataset_name in DATASET_NAMES
    x_train = np.load(DATA_DIR + dataset_name + '/benign/x_train.npy')
    y_train = np.load(DATA_DIR + dataset_name + '/benign/y_train.npy')
    x_test = np.load(DATA_DIR + dataset_name + '/benign/x_test.npy')
    y_test = np.load(DATA_DIR + dataset_name + '/benign/y_test.npy')
    return x_train, y_train, x_test, y_test


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


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']