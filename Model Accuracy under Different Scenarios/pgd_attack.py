from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import ProjectedGradientDescent

import tensorflow as tf
import os


####for solving some specific problems, don't care
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import keras
from keras import backend as K
import numpy as np
import time
from util import get_model
import sys
import argparse


# def JSMA(model, x, y):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     jsma = SaliencyMapMethod(model_wrap, sess=sess)
#     jsma_params = {'theta':1., 'gamma': 0.1, 'clip_min':0., 'clip_max':1.}
#     adv = jsma.generate_np(x, **jsma_params)
#     return adv
#
#
def FGSM(model, x, y):
    sess = K.get_session()
    model_wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(model_wrap, sess=sess)
    fgsm_params={'y':y, 'eps':0.2, 'clip_min':0., 'clip_max': 1.}
    adv = fgsm.generate_np(x, **fgsm_params)
    return adv
#
#
# def BIM(model, x, y):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     bim = BasicIterativeMethod(model_wrap, sess=sess)
#     bim_params={'eps_iter': 0.03, 'nb_iter': 10, 'y':y, 'clip_min': 0., 'clip_max': 1.}
#     adv = bim.generate_np(x, **bim_params)
#     return adv

# # invoke the method for many times leads to multiple symbolic graph and may cause OOM
# def CW(model, x, y, batch):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     cw = CarliniWagnerL2(model_wrap, sess=sess)
#     cw_params = {'binary_search_steps': 1,
#                  'y': y,
#                  'learning_rate': .1,
#                  'max_iterations': 50,
#                  'initial_const': 10,
#                  'batch_size': batch,
#                  'clip_min': -0.5,
#                  'clip_max': 0.5}
#     adv = cw.generate_np(x, **cw_params)# invoke the method for many times leads to multiple symbolic graph and may cause OOM
# # def CW(model, x, y, batch):
# #     sess = K.get_session()
# #     model_wrap = KerasModelWrapper(model)
# #     cw = CarliniWagnerL2(model_wrap, sess=sess)
# #     cw_params = {'binary_search_steps': 1,
# #                  'y': y,
# #                  'learning_rate': .1,
# #                  'max_iterations': 50,
# #                  'initial_const': 10,
# #                  'batch_size': batch,
# #                  'clip_min': -0.5,
# #                  'clip_max': 0.5}
# #     adv = cw.generate_np(x, **cw_params)
# #     return adv
# #
# #
# # # for mnist, eps=.3, eps_iter=.03, nb_iter=10
# # # for cifar and svhn, eps=8/255, eps_iter=.01, nb_iter=30
# # def PGD(model, x, y, batch):
# #     sess = K.get_session()
# #     model_wrap = KerasModelWrapper(model)
# #     pgd = ProjectedGradientDescent(model_wrap, sess=sess)
# #     pgd_params = {'eps': 8. / 255.,
# #                   'eps_iter': .01,
# #                   'nb_iter': 30.,
# #                   'clip_min': -0.5,
# #                   'clip_max': 0.5,
# #                   'y': y}
# #     adv = pgd.generate_np(x, **pgd_params)
# #     return adv
#     return adv
#
#
# # for mnist, eps=.3, eps_iter=.03, nb_iter=10
# # for cifar and svhn, eps=8/255, eps_iter=.01, nb_iter=30
# def PGD(model, x, y, batch):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     pgd = ProjectedGradientDescent(model_wrap, sess=sess)
#     pgd_params = {'eps': 8. / 255.,
#                   'eps_iter': .01,
#                   'nb_iter': 30.,
#                   'clip_min': -0.5,
#                   'clip_max': 0.5,
#                   'y': y}
#     adv = pgd.generate_np(x, **pgd_params)
#     return adv


# integrate all attack method in one function and only construct graph once
def gen_adv_data(model, x, y, method, dataset, batch=2048):
    sess = K.get_session()
    model_wrap = KerasModelWrapper(model)
    if method.upper() == 'CW':
        params = {'binary_search_steps': 1,
                  'y': y,
                  'learning_rate': .1,
                  'max_iterations': 50,
                  'initial_const': 10,
                  'batch_size': batch,
                  # 'clip_min': -0.5,
                  # 'clip_max': 0.5
                  }
        attack = CarliniWagnerL2(model_wrap, sess=sess)

    elif method.upper() == 'PGD':
        # set parameters according to dataset
        if dataset == 'cifar':
            params = {'eps': 16. / 255.,
                      'eps_iter': 2. / 255.,
                      'nb_iter': 30.,
                      # 'clip_min': -0.5,
                      # 'clip_max': 0.5,
                      'y': y}
        elif dataset == 'mnist':
            params = {'eps': .3,
                      'eps_iter': .03,
                      'nb_iter': 20.,
                      'clip_min': -0.5,
                      'clip_max': 0.5,
                      'y': y}
        elif dataset == 'svhn':
            params = {'eps': 8. / 255, # what about make it lager, original 8. / 255
                      'eps_iter': 0.01,
                      'nb_iter': 30.,
                      'clip_min': -0.5,
                      'clip_max': 0.5,
                      'y': y}
        attack = ProjectedGradientDescent(model_wrap, sess=sess)
    elif method.upper() == 'FGSM':
        params={'y':y, 'eps':0.2, 'clip_min':-0.5, 'clip_max': 0.5}
        # params = {'eps': 8. / 255.,
        #             'clip_min': -0.5,
        #             'clip_max': 0.5,
        #             'y': y}
        attack = FastGradientMethod(model_wrap, sess=sess)


    else:
        print('Unsupported attack')
        sys.exit(1)


    data_num = x.shape[0]
    begin, end = 0, batch
    adv_x_all = np.zeros_like(x)
    # every time process batch_size
    while end < data_num:
        start_time = time.time()
        params['y'] = y[begin:end]
        adv_x = attack.generate_np(x[begin:end], **params)
        adv_x_all[begin: end] = adv_x
        print(begin, end, "done")
        begin += batch
        end += batch
        end_time = time.time()
        print("time: ", end_time - start_time)

    # process the remaining
    if begin < data_num:
        start_time = time.time()
        params['y'] = y[begin:]
        adv_x = attack.generate_np(x[begin:], **params)
        adv_x_all[begin:] = adv_x
        print(begin, data_num, "done")
        end_time = time.time()
        print("time: ", end_time - start_time)

    return adv_x_all



# the data is in range(-.5, .5)
def load_data(name):
    assert (name.upper() in ['MNIST', 'CIFAR', 'SVHN'])
    name = name.lower()
    x_train = np.load('./data/' + name + '_data/' + name + '_x_train.npy')
    y_train = np.load('./data/' + name + '_data/' + name + '_y_train.npy')
    x_test = np.load('./data/' + name + '_data/' + name + '_x_test.npy')
    y_test = np.load('./data/' + name + '_data/' + name + '_y_test.npy')
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
    print(np.sum(idx) / num)


if __name__ == '__main__':

    datasets = ['mnist', 'svhn'] # , 'cifar'
    model_dict = {
                'mnist': ['lenet1', 'lenet4', 'lenet5'],
                'cifar': ['vgg16'], # , 'resnet20'
                'svhn' : ['svhn_model', 'svhn_second', 'svhn_first']
                }
    attacks = ['FGSM', 'PGD']
    for attack in attacks: 
        for dataset in datasets:
            for model_name in model_dict[dataset]:
                print(">>>>>>>>>>>>>>>>>>   " + attack)
                print(">>>>>>>>>>>>>>>>>>   " + dataset)
                print(">>>>>>>>>>>>>>>>>>   " + model_name)
                x_train, y_train, x_test, y_test = load_data(dataset)

                # ## load deephunter model
                from keras.models import load_model
                model = load_model('new_model/dp_{}.h5'.format(model_name))
                model.summary()

                accuracy(model, x_test, y_test)

                adv = gen_adv_data(model, x_test, y_test, attack, dataset, 256)

                np.save('./data/' + dataset + '_data/model/' + model_name + '_' + attack + '.npy', adv)

