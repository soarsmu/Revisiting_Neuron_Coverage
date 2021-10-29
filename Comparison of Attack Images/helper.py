import numpy as np

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


# the data is in range(-.5, .5)
def load_data(dataset_name):
    assert dataset_name in DATASET_NAMES
    x_train = np.load(DATA_DIR + dataset_name + '/benign/x_train.npy')
    y_train = np.load(DATA_DIR + dataset_name + '/benign/y_train.npy')
    x_test = np.load(DATA_DIR + dataset_name + '/benign/x_test.npy')
    y_test = np.load(DATA_DIR + dataset_name + '/benign/y_test.npy')
    return x_train, y_train, x_test, y_test

def load_adv_data(dataset, model, attack):
    adv_dir = "{}{}/adv/{}/{}/".format(DATA_DIR, dataset, model, attack)
    x_adv_path = "{}x_test.npy".format(adv_dir)
    y_adv_path = "{}y_test.npy".format(adv_dir)
    x_adv = np.load(x_adv_path)
    y_adv = np.load(y_adv_path)
    return x_adv, y_adv

