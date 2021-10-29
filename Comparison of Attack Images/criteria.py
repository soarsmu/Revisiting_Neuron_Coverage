import numpy as np
import tensorflow as tf
import os
from attack_evaluate import AttackEvaluate
import argparse

import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from helper import load_data, load_adv_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_DIR = "../models/"
####for solving some specific problems, don't care
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

def lp_distance(pic1, pic2):
    pic1 = np.round(pic1*255)
    pic2 = np.round(pic2*255)


    dist_li = np.linalg.norm(np.reshape(pic1 - pic2, -1), ord=np.inf)

    return dist_li

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MR and Linf')
    parser.add_argument('-dataset', help="dataset to use", choices=['mnist', 'cifar', 'svhn'])
    parser.add_argument('-model', help="target model to attack",
                        choices=['vgg16', 'resnet20', 'lenet1', 'lenet4', 'lenet5', 'svhn_model',
                                 'svhn_first', 'svhn_second'])

    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model
    # print(args.dataset)


    ### load deephunter adv
    x_adv_deep, y_dp = load_adv_data(dataset, model_name, 'deephunter')
    ### load PGD adv
    x_adv_pgd, y_dg = load_adv_data(dataset, model_name, 'pgd')
    for i in range(len(y_dp)):
        assert y_dp[i].all() == y_dp[i].all()
    # load dataset
    x_train, y_train, x_test, y_test = load_data(dataset)

    # import model
    from keras.models import load_model
    model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset, model_name)
    model = load_model(model_path)
    model.summary()


    ## compare MR
    criteria1 = AttackEvaluate(model, x_test, y_test, x_adv_deep[:1000])
    MR_deep = criteria1.misclassification_rate()
    print(MR_deep)

    criteria2 = AttackEvaluate(model, x_test, y_test, x_adv_pgd[:1000])
    MR_pgd = criteria2.misclassification_rate()
    print(MR_pgd)

    li_deep = 0
    li_pgd = 0

    num = 0

    for i in range(1000):
        if softmax(model.predict(np.expand_dims(x_adv_deep[i], axis=0))).argmax(axis=-1) != softmax(model.predict(np.expand_dims(x_test[i], axis=0))).argmax(axis=-1):
            num += 1

            ## compare Lp
            li_deep_new = lp_distance(x_test[i], x_adv_deep[i])
            li_pgd_new = lp_distance(x_test[i], x_adv_pgd[i])

            li_deep += li_deep_new
            li_pgd += li_pgd_new

    if dataset == 'cifar':
        li_deep = li_deep/255

    os.makedirs('./table_2/{}/{}'.format(dataset, model_name), exist_ok=True)
    with open("./table_2/{}/{}/compare_result.txt".format(dataset, model_name), "a") as f:
        f.write("\n------------------------------------------------------------------------------\n")
        f.write('the result of {} {} is: \n'.format(args.dataset, args.model))
        f.write('MR of DH is {} \n'.format(MR_deep))
        f.write('MR of DP is {} \n'.format(MR_pgd))
        f.write('Linf of DH is {} \n'.format(li_deep/num))
        f.write('Linf of DP is {} \n'.format(li_pgd/num))


    print("li of deephunter is: {}".format(li_deep/num))
    print("li of pgd is: {}".format(li_pgd/num))




