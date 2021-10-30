import argparse
import os
import random
import shutil
import warnings
import sys

warnings.filterwarnings("ignore")

from keras import backend as K
import numpy as np
from PIL import Image, ImageFilter
from skimage.measure import compare_ssim as SSIM
import keras

import tensorflow as tf
import os
from helper import load_data
from helper import Coverage


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATA_DIR = "../data/"
MODEL_DIR = "../models/"
RESULT_DIR = "../coverage/"

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
ST = "spatialtransformation"
ATTACK_NAMES = [APGD, BIM, CW, DF, FGSM, JSMA, NF, PGD, SA, ST]

# helper function


if __name__ == '__main__':
    dataset = 'mnist'
    model_name = 'lenet1'
    l = [0, 8]

    x_train, y_train, x_test, y_test = load_data(dataset)

    # ## load mine trained model
    from keras.models import load_model
    model_path = "{}{}/{}.h5".format(MODEL_DIR,
                                    dataset, model_name)
    model = load_model(model_path)

    model.summary()

    tknp_all = np.array([])

    for num in range(0, 50):
        coverage = Coverage(model, x_train, y_train, x_test, y_test, x_test[0: 200*num])
        tknp = coverage.TKNP(l)
        tknp_all = np.append(tknp_all, tknp)

        with open("testing_coverage_result.txt", "a") as f:
            f.write("\n------------------------------------------------------------------------------\n")
            f.write('x: {}   \n'.format(num*200+1))
            f.write('TKNP: {} \n'.format(tknp))

    np.save('Q2_original/tknp_all.npy', tknp_all)
