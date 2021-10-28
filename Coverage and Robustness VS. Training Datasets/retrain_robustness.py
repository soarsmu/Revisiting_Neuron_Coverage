import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import os
from helper import AttackEvaluate
from helper import load_data, retrain

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
    order_number = args.order_number

    l = [0, model_layer]

    os.makedirs("attack_results", exist_ok=True)

    x_train, y_train, x_test, y_test = load_data(dataset)
    x_test_new = np.load('x_test_new.npy')

    # ## load mine trained model
    from keras.models import load_model

    model = load_model('../data/' + dataset + '_data/model/' + model_name + '.h5')
    model.summary()

    T = 1

    for i in range(T):
        index = np.load('fuzzing/nc_index_{}.npy'.format(i), allow_pickle=True).item()
        for y, x in index.items():
            x_train = np.concatenate((x_train, np.expand_dims(x, axis=0)), axis=0)
            y_train = np.concatenate((y_train, np.expand_dims(y_train[y], axis=0)), axis=0)

    retrained_model = retrain(model, x_train, y_train, x_test, y_test, batch_size=512, epochs=30)
    retrained_model.save('new_model/' + dataset +'/model_{}.h5'.format(T-1))

    criteria = AttackEvaluate(retrained_model, x_test, y_test, x_test_new)

    MR = criteria.misclassification_rate()
    ACAC = criteria.avg_confidence_adv_class()
    ACTC = criteria.avg_confidence_true_class()
    ALP_L0, ALP_L2, ALP_Li = criteria.avg_lp_distortion()
    ASS = criteria.avg_SSIM()
    PSD = criteria.avg_PSD()
    NTE = criteria.avg_noise_tolerance_estimation()
    _, _, RGB = criteria.robust_gaussian_blur()
    _, _, RIC = criteria.robust_image_compression(1)

    with open("attack_results/attack_evaluate_result_{}.txt".format(T), "a") as f:
        f.write("\n------------------------------------------------------------------------------\n")
        f.write('the result of {} {} is: \n'.format(dataset, model_name))
        f.write('MR: {} \n'.format(MR))
        f.write('ACAC: {} \n'.format(ACAC))
        f.write('ACTC: {} \n'.format(ACTC))
        f.write('ALP_L0: {} \n'.format(ALP_L0))
        f.write('ALP_L2: {} \n'.format(ALP_L2))
        f.write('ALP_Li: {} \n'.format(ALP_Li))
        f.write('ASS: {} \n'.format(ASS))
        f.write('PSD: {} \n'.format(PSD))
        f.write('NTE: {} \n'.format(NTE))
        f.write('RGB: {} \n'.format(RGB))
        f.write('RIC: {} \n'.format(RIC))