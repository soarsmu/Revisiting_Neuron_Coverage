import argparse
import warnings

warnings.filterwarnings("ignore")

from keras import backend as K
import numpy as np
from helper import Coverage
from helper import load_data
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

####for solving some specific problems, don't care
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='lenet1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--model_layer", default=8, type=int)
    parser.add_argument("--order_number", default=1, type=int)

    args = parser.parse_args()
    model_name = args.model_name
    model_layer = args.model_layer
    dataset = args.dataset
    T = args.order_number

    os.makedirs("coverage_coverage_results", exist_ok=True)
    l = [0, model_layer]

    x_train, y_train, x_test, y_test = load_data(dataset)

    # ## load mine trained model
    from keras.models import load_model

    model = load_model('../data/' + dataset + '_data/model/' + model_name + '.h5')
    model.summary()

    x_train_new = x_train

    for i in range(T):
        index = np.load('fuzzing/nc_index_{}.npy'.format(i), allow_pickle=True).item()
        for y, x in index.items():
            x_train_new = np.concatenate((x_train_new, np.expand_dims(x, axis=0)), axis=0)

    coverage = Coverage(model, x_train, y_train, x_test, y_test, x_train_new)
    nc1, activate_num1, total_num1 = coverage.NC(l, threshold=0.1)
    nc2, activate_num2, total_num2 = coverage.NC(l, threshold=0.3)
    nc3, activate_num3, total_num3 = coverage.NC(l, threshold=0.5)
    nc4, activate_num4, total_num4 = coverage.NC(l, threshold=0.7)
    nc5, activate_num5, total_num5 = coverage.NC(l, threshold=0.9)
    tknc, pattern_num, total_num6 = coverage.TKNC(l)
    tknp = coverage.TKNP(l)
    kmnc, nbc, snac, covered_num, l_covered_num, u_covered_num, neuron_num = coverage.KMNC(l)

    with open("coverage_coverage_results/coverage_result_{}.txt".format(T), "a") as f:
        f.write("\n------------------------------------------------------------------------------\n")
        f.write('the result of {} {}is: \n'.format(dataset, model_name))
        f.write('NC(0.1): {}  activate_num: {}  total_num: {} \n'.format(nc1, activate_num1, total_num1))
        f.write('NC(0.3): {}  activate_num: {}  total_num: {} \n'.format(nc2, activate_num2, total_num2))
        f.write('NC(0.5): {}  activate_num: {}  total_num: {} \n'.format(nc3, activate_num3, total_num3))
        f.write('NC(0.7): {}  activate_num: {}  total_num: {} \n'.format(nc4, activate_num4, total_num4))
        f.write('NC(0.9): {}  activate_num: {}  total_num: {} \n'.format(nc5, activate_num5, total_num5))
        f.write('TKNC: {}  pattern_num: {}  total_num: {} \n'.format(tknc, pattern_num, total_num6))
        f.write('TKNP: {} \n'.format(tknp))
        f.write('KMNC: {}  covered_num: {}  total_num: {} \n'.format(kmnc, covered_num, neuron_num))
        f.write('NBC: {}  l_covered_num: {}  u_covered_num: {} \n'.format(nbc, l_covered_num, u_covered_num))
        f.write('SNAC: {} \n'.format(snac))

