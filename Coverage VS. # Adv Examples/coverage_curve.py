import argparse
import os
import random
import shutil
import warnings
import sys

import logging
import time
from datetime import datetime
import pytz
import numpy as np

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


import numpy as np
from PIL import Image, ImageFilter
from skimage.measure import compare_ssim as SSIM
import keras
from keras.models import load_model
from helper import load_data
from helper import Coverage
import tensorflow as tf
import os

## custom time zone for logger
def customTime(*args):
    utc_dt = pytz.utc.localize(datetime.utcnow())
    converted = utc_dt.astimezone(pytz.timezone("Singapore"))
    return converted.timetuple()

logging.Formatter.converter = customTime


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VERBOSE = False

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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Attack for DNN')
    parser.add_argument(
        '--dataset', help="Model Architecture", type=str, default="mnist")
    parser.add_argument(
        '--model', help="Model Architecture", type=str, default="lenet1")
    parser.add_argument(
        '--attack', help="Adversarial examples", type=str, default="fgsm")
    
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    attack_name = args.attack

    ## Prepare directory for loading adversarial images and logging
    adv_dir = "{}{}/adv/{}/{}/".format(
        DATA_DIR, dataset_name, model_name, attack_name)
    cov_dir = "{}{}/adv/{}/{}/".format(
        RESULT_DIR, dataset_name, model_name, attack_name)
    
    if not os.path.exists(cov_dir):
            os.makedirs(cov_dir)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(cov_dir, 'output.log')),
            logging.StreamHandler()
        ])

    ## Load benign images from mnist, cifar, or svhn
    x_train, y_train, x_test, y_test = load_data(dataset_name)

    ## Load keras pretrained model for the specific dataset
    model_path = "{}{}/{}.h5".format(MODEL_DIR,
                                    dataset_name, model_name)
    model = load_model(model_path)
    model.summary()

    x_adv_path = "{}x_test.npy".format(adv_dir)
    x_adv = np.load(x_adv_path)

    l = [0, 8]

    xlabel = []
    cov_nc1 = []
    cov_nc2 = []
    cov_kmnc = []
    cov_nbc = []
    cov_snac = []
    cov_tknc = []
    cov_tknp = []

    cov_result_path = os.path.join(cov_dir, "coverage_result.txt")
    with open(cov_result_path, "w+") as f:
        for i in range(1, len(x_adv), 200):
            if i == 1000 or i == 3000 or i == 5000 or i == 7000 or i == 9000:
                print(i)

            coverage = Coverage(model, x_train, y_train, x_test, y_test, x_adv[:i])
            nc1, _, _ = coverage.NC(l, threshold=0.3)
            nc2, _, _ = coverage.NC(l, threshold=0.5)
            kmnc, nbc, snac, _, _, _, _ = coverage.KMNC(l)
            tknc, _, _ = coverage.TKNC(l)
            tknp = coverage.TKNP(l)

            f.write("\n------------------------------------------------------------------------------\n")
            f.write('x: {}   \n'.format(i))
            f.write('NC(0.1): {}   \n'.format(nc1))
            f.write('NC(0.3): {}   \n'.format(nc2))
            f.write('TKNC: {}   \n'.format(tknc))
            f.write('TKNP: {} \n'.format(tknp))
            f.write('KMNC: {} \n'.format(kmnc))
            f.write('NBC: {}  \n'.format(nbc))
            f.write('SNAC: {} \n'.format(snac))

            xlabel.append(i)
            cov_nc1.append(nc1)
            cov_nc2.append(nc2)
            cov_kmnc.append(kmnc)
            cov_nbc.append(nbc)
            cov_snac.append(snac)
            cov_tknc.append(tknc)
            cov_tknp.append(tknp)

        np.save(os.path.join(cov_dir, 'xlabel.npy'), xlabel)
        np.save(os.path.join(cov_dir, 'cov_nc1.npy'), cov_nc1)
        np.save(os.path.join(cov_dir, 'cov_nc2.npy'), cov_nc2)
        np.save(os.path.join(cov_dir, 'cov_kmnc.npy'), cov_kmnc)
        np.save(os.path.join(cov_dir, 'cov_nbc.npy'), cov_nbc)
        np.save(os.path.join(cov_dir, 'cov_snac.npy'), cov_snac)
        np.save(os.path.join(cov_dir, 'cov_tknc.npy'), cov_tknc)
        np.save(os.path.join(cov_dir, 'cov_tknp.npy'), cov_tknp)



