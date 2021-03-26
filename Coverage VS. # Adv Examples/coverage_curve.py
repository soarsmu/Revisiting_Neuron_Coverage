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

from keras import backend as K
import numpy as np
from PIL import Image, ImageFilter
from skimage.measure import compare_ssim as SSIM
import keras
from keras.models import load_model

from util import get_model

import tensorflow as tf
import os

## custom time zone for logger
def customTime(*args):
    utc_dt = pytz.utc.localize(datetime.utcnow())
    converted = utc_dt.astimezone(pytz.timezone("Singapore"))
    return converted.timetuple()

logging.Formatter.converter = customTime
logger = logging.getLogger("coverage_criteria")


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


####for solving some specific problems, don't care
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# helper function
def get_layer_i_output(model, i, data):
    layer_model = K.function([model.layers[0].input], [model.layers[i].output])
    ret = layer_model([data])[0]
    num = data.shape[0]
    ret = np.reshape(ret, (num, -1))
    return ret

# the data is in range(-.5, .5)
def load_data(dataset_name):
    assert dataset_name in DATASET_NAMES
    x_train = np.load(DATA_DIR + dataset_name + '/benign/x_train.npy')
    y_train = np.load(DATA_DIR + dataset_name + '/benign/y_train.npy')
    x_test = np.load(DATA_DIR + dataset_name + '/benign/x_test.npy')
    y_test = np.load(DATA_DIR + dataset_name + '/benign/y_test.npy')
    return x_train, y_train, x_test, y_test


class Coverage:
    def __init__(self, model, x_train, y_train, x_test, y_test, x_adv):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_adv = x_adv
    
    # find scale factors and min num
    def scale(self, layers, batch=1024):
        data_num = self.x_adv.shape[0]
        factors = dict()
        for i in layers:
            begin, end = 0, batch
            max_num, min_num = np.NINF, np.inf 
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                tmp = layer_output.max()
                max_num = tmp if tmp > max_num else max_num
                tmp = layer_output.min()
                min_num = tmp if tmp < min_num else min_num
                begin += batch
                end += batch
            factors[i] = (max_num - min_num, min_num)
        return factors


    #1 Neuron Coverage
    def NC(self, layers, threshold=0., batch=1024):
        factors = self.scale(layers, batch=batch)
        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)
        
        activate_num = 0
        data_num = self.x_adv.shape[0]
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            buckets = np.zeros(neurons).astype('bool')
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                # scale the layer output to (0, 1)
                layer_output -= factors[i][1]
                layer_output /= factors[i][0]
                col_max = np.max(layer_output, axis=0)
                begin += batch
                end += batch
                buckets[col_max > threshold] = True
            activate_num += np.sum(buckets)
        # logger.info('NC:\t{:.3f} activate_num:\t{} neuron_num:\t{}'.format(activate_num / neuron_num, activate_num, neuron_num))
        return activate_num / neuron_num, activate_num, neuron_num

    #2 k-multisection neuron coverage, neuron boundary coverage and strong activation neuron coverage
    def KMNC(self, layers, k=10, batch=1024):
        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)
        
        covered_num = 0
        l_covered_num = 0
        u_covered_num = 0
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            logger.info(neurons)
            begin, end = 0, batch
            data_num = self.x_train.shape[0]
            
            neuron_max = np.full(neurons, np.NINF).astype('float')
            neuron_min = np.full(neurons, np.inf).astype('float')
            while begin < data_num:
                layer_output_train = get_layer_i_output(self.model, i, self.x_train[begin:end])
                batch_neuron_max = np.max(layer_output_train, axis=0)
                batch_neuron_min = np.min(layer_output_train, axis=0)
                neuron_max = np.maximum(batch_neuron_max, neuron_max)
                neuron_min = np.minimum(batch_neuron_min, neuron_min)
                begin += batch
                end += batch
            buckets = np.zeros((neurons, k + 2)).astype('bool')
            interval = (neuron_max - neuron_min) / k
            # logger.info(interval[8], neuron_max[8], neuron_min[8])
            begin, end = 0, batch
            data_num = self.x_adv.shape[0]
            while begin < data_num:
                layer_output_adv = get_layer_i_output(model, i, self.x_adv[begin: end])
                layer_output_adv -= neuron_min
                layer_output_adv /= (interval + 10**(-100))
                layer_output_adv[layer_output_adv < 0.] = -1
                layer_output_adv[layer_output_adv >= k/1.0] = k
                layer_output_adv = layer_output_adv.astype('int')
                # index 0 for lower, 1 to k for between, k + 1 for upper
                layer_output_adv = layer_output_adv + 1
                for j in range(neurons):
                    uniq = np.unique(layer_output_adv[:, j])
                    # logger.info(layer_output_adv[:, j])
                    buckets[j, uniq] = True
                begin += batch
                end += batch
            covered_num += np.sum(buckets[:,1:-1])
            u_covered_num += np.sum(buckets[:, -1])
            l_covered_num += np.sum(buckets[:, 0])
        logger.info('KMNC:\t{:.3f} covered_num:\t{}'.format(covered_num / (neuron_num * k), covered_num))
        logger.info('NBC:\t{:.3f} l_covered_num:\t{}'.format((l_covered_num + u_covered_num) / (neuron_num * 2), l_covered_num))
        logger.info('SNAC:\t{:.3f} u_covered_num:\t{}'.format(u_covered_num / neuron_num, u_covered_num))
        return covered_num / (neuron_num * k), (l_covered_num + u_covered_num) / (neuron_num * 2), u_covered_num / neuron_num, covered_num, l_covered_num, u_covered_num, neuron_num*k

    #3 top-k neuron coverage
    def TKNC(self,layers, k=2, batch=1024):
        def top_k(x, k):
            ind = np.argpartition(x, -k)[-k:]
            return ind[np.argsort((-x)[ind])]

        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)
        
        pattern_num = 0
        data_num = self.x_adv.shape[0]
        for i in layers:
            pattern_set = set()
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                topk = np.argpartition(layer_output, -k, axis=1)[:, -k:]
                topk = np.sort(topk, axis=1)
                # or in order
                #topk = np.apply_along_axis[lambda x: top_k(layer_output, k), 1, layer_output]
                for j in range(topk.shape[0]):
                    for z in range(k):
                        pattern_set.add(topk[j][z])
                begin += batch
                end += batch
            pattern_num += len(pattern_set)
        logger.info('TKNC:\t{:.3f} pattern_num:\t{} neuron_num:\t{}'.format(pattern_num / neuron_num, pattern_num, neuron_num))
        return pattern_num / neuron_num, pattern_num, neuron_num

    #4 top-k neuron patterns 
    def TKNP(self, layers, k=2, batch=1024):
        def top_k(x, k):
            ind = np.argpartition(x, -k)[-k:]
            return ind[np.argsort((-x)[ind])]
        def to_tuple(x):
            l = list()
            for row in x:
                l.append(tuple(row))
            return tuple(l)

        pattern_set = set()
        layer_num = len(layers)
        data_num = self.x_adv.shape[0]
        patterns = np.zeros((data_num, layer_num, k))
        layer_cnt = 0
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                topk = np.argpartition(layer_output, -k, axis=1)[:, -k:]
                topk = np.sort(topk, axis=1)
                # or in order
                #topk = np.apply_along_axis[lambda x: top_k(layer_output, k), 1, layer_output]
                patterns[begin:end, layer_cnt, :] = topk
                begin += batch
                end += batch
            layer_cnt += 1
        
        for i in range(patterns.shape[0]):
            pattern_set.add(to_tuple(patterns[i]))
        pattern_num = len(pattern_set)
        logger.info('TKNP:\t{:.3f}'.format(pattern_num))
        return pattern_num
    
    def all(self, layers, batch=100):
        self.NC(layers, batch=batch)
        self.KMNC(layers, batch=batch)
        self.TKNC(layers, batch=batch)
        self.TKNP(layers, batch=batch)

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
                logger.info(i)

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
            logger.info(xlabel)

        np.save(os.path.join(cov_dir, 'xlabel.npy'), xlabel)
        np.save(os.path.join(cov_dir, 'cov_nc1.npy'), cov_nc1)
        np.save(os.path.join(cov_dir, 'cov_nc2.npy'), cov_nc2)
        np.save(os.path.join(cov_dir, 'cov_kmnc.npy'), cov_kmnc)
        np.save(os.path.join(cov_dir, 'cov_nbc.npy'), cov_nbc)
        np.save(os.path.join(cov_dir, 'cov_snac.npy'), cov_snac)
        np.save(os.path.join(cov_dir, 'cov_tknc.npy'), cov_tknc)
        np.save(os.path.join(cov_dir, 'cov_tknp.npy'), cov_tknp)



