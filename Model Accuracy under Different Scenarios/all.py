import sys
sys.path.append('..')
import parameters as param
from utils import load_data

import argparse
import os

import random
import shutil
import warnings

warnings.filterwarnings("ignore")

from keras import backend as K
import numpy as np
from PIL import Image, ImageFilter
from skimage.measure import compare_ssim as SSIM
import keras
from util import get_model

import tensorflow as tf
import os
import prettytable as pt
from attack import gen_adv_data
from keras.models import load_model
from helper import check_data_path

class AttackEvaluate:
    # model does not have softmax layer
    def __init__(self, model, ori_x, ori_y, adv_x):
        self.model = model
        # get the raw data
        self.nature_samples = ori_x
        self.labels_samples = ori_y
        # get the adversarial examples
        self.adv_samples = adv_x
        
        predictions = model.predict(self.adv_samples)

        def soft_max(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        tmp_soft_max = []
        for i in range(len(predictions)):
            tmp_soft_max.append(soft_max(predictions[i]))

        self.softmax_prediction = np.array(tmp_soft_max)

    # help function
    def successful(self, adv_softmax_preds, nature_true_preds):
        if np.argmax(adv_softmax_preds) != np.argmax(nature_true_preds):
            return True
        else:
            return False

    # 1 MR:Misclassification Rate
    def misclassification_rate(self):

        cnt = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
        mr = cnt / len(self.adv_samples)
        print('MR:\t\t{:.1f}%'.format(mr * 100))
        return mr



if __name__ == '__main__':
    
    datasets = ['cifar', 'mnist', 'svhn']
    model_dict = {
                'mnist': ['lenet1', 'lenet4', 'lenet5'],
                'cifar': ['vgg16', 'resnet20'],
                'svhn' : ['svhn_model', 'svhn_second', 'svhn_first']
                }

    # datasets = ['cifar', 'eurosat']
    # model_dict = {
    #     'cifar': ['resnet56'],
    #     'eurosat': ['resnet20', 'resnet56'],
    # }

    defense_names = [param.BENIGN, param.DEEPHUNTER]
    optim_defenses = [param.FGSM, param.PGD, param.APGD]
    
    attack_names = [param.BENIGN, param.DEEPHUNTER]
    optim_attacks = [param.FGSM, param.PGD, param.APGD]
    
    table = pt.PrettyTable()
    table.field_names = ["Dataset", "Model"] + attack_names + optim_attacks
    ### Set align
    for field_name in ["Dataset", "Model"] + attack_names + optim_attacks:
        table.align[field_name] = 'l'

    for dataset_name in datasets:

        ### benign dataset
        x_train, y_train, x_test, y_test = load_data(dataset_name)

        for model_name in model_dict[dataset_name]:

            '''Load models for denfense'''
            model_defenses = {}
            for defense in defense_names + optim_defenses:
                # To-Do: to be modified after we generate using new paths.    
                if defense == param.BENIGN:
                    ### model path for benign model
                    model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
                elif defense == param.DEEPHUNTER:
                    ### model path for deephunter model
                    model_path = '{}{}/adv_{}_deephunter.h5'.format(param.MODEL_DIR, dataset_name, model_name)
                else:
                    ### model path for models trained with optimization-based attack
                    model_path = "{}{}/adv_{}_{}.h5".format(param.MODEL_DIR, dataset_name, model_name, defense)
                
                model_defenses[defense] = load_model(model_path)
            
            '''Load dataset'''
            x_adv_attacks = {}
            for attack in attack_names:
                # To-Do: to be modified after we generate using new paths.
                if attack == param.BENIGN:
                    x_adv_attacks[attack] = x_test
                elif attack == param.DEEPHUNTER:
                    ### deephunter dataset
                    adv_dir = "{}{}/adv/{}/{}/".format(param.DATA_DIR, dataset_name, model_name, param.DEEPHUNTER)
                    dp_adv_path = "{}x_test.npy".format(adv_dir)
                    x_adv_attacks[attack] = np.load(dp_adv_path)
                else:
                    pass

            '''Computing accuracy'''

            ### get benign data
            defense = param.BENIGN
            benign_content = {}
            for attack in attack_names + optim_attacks:
                
                if  (attack in optim_attacks):
                    # TODO:  only generate adv examples when the saved data is not exist
                    adv_example_fpath = ""
                    # generate adv examples
                    adv_examples = gen_adv_data(model_defenses[defense], x_test, y_test, attack, dataset_name, 256)
                elif (attack in attack_names):
                    adv_examples = x_adv_attacks[attack]

                criteria = AttackEvaluate(model_defenses[defense], x_test, y_test, adv_examples)
                accuracy = 1 - criteria.misclassification_rate()
                benign_content[attack] = accuracy

            for defense in defense_names + optim_defenses:
                row_content = [dataset_name, model_name + '_' + defense]
                
                for attack in attack_names + optim_attacks:
                    if  (attack in optim_attacks):
                        
                        # generate adv examples
                        adv_examples = gen_adv_data(model_defenses[defense], x_test, y_test, attack, dataset_name, 256)
                    
                    elif (attack in attack_names):
                        adv_examples = x_adv_attacks[attack]

                    criteria = AttackEvaluate(model_defenses[defense], x_test, y_test, adv_examples)
                    accuracy = 1 - criteria.misclassification_rate()
                    difference = accuracy - benign_content[attack]

                    row_content.append(str(round(accuracy * 100,2)) + 
                                        '(' + str(round(difference * 100,2)) + ')')
                
                table.add_row(row_content)
            
            
            # TODO: save table into external files
            print(table)

