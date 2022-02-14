import sys
sys.path.append('..')
import parameters as param
from utils import load_data

import argparse
import os
import sys, logging
import time
from datetime import datetime
import pytz
import numpy as np
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
        # print('MR:\t\t{:.1f}%'.format(mr * 100))
        return mr

## custom time zone for logger
def customTime(*args):
    utc_dt = pytz.utc.localize(datetime.utcnow())
    converted = utc_dt.astimezone(pytz.timezone("Singapore"))
    return converted.timetuple()


if __name__ == '__main__':


    os.makedirs("missclassification_rate/", exist_ok=True)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join("missclassification_rate/", 'output.log')),
            logging.StreamHandler()
        ])

    logging.Formatter.converter = customTime
    logger = logging.getLogger("missclassification_rate")
        
   
    ##
    #  Use this for generating table 3
    ## 
    
    model_dict = {
            'mnist': ['lenet5'],
            'cifar': ['vgg16'],
            'svhn' : ['svhn_second'],
            'eurosat': ['resnet56']
            }

    logger.info("\nGenerating Table 3. Misclassification rates for defended models")
    
    defense_names = [param.BENIGN, param.DIFFERENTIABLE, param.NONDIFFERENTIABLE, param.DEEPHUNTER, param.FGSM, param.PGD ]
    attack_names = [param.BENIGN, param.DIFFERENTIABLE, param.NONDIFFERENTIABLE, param.DEEPHUNTER, param.FGSM, param.PGD]
    metric = "missclassification_rate"
    

    # Check models and adv examples 
    # to ensure that we can load models and adv examples

    for dataset_name in model_dict.keys():

        for model_name in model_dict[dataset_name]:

            for defense in defense_names :
    
                '''check models for defense'''
                if defense == param.BENIGN:
                    ### model path for benign model
                    model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
                else :
                    ### model path for adversarially trained model
                    model_path = "{}{}/adv_{}_{}.h5".format(param.MODEL_DIR, dataset_name, model_name, defense)
                
                if not os.path.exists(model_path) :
                    raise ValueError(f"{model_path} model does not exist")
                
                '''check dataset and adv examples'''
                if defense == param.BENIGN:
                    defended_model_dir = "{}{}/adv/{}".format(param.DATA_DIR, dataset_name, model_name)
                else :
                    defended_model_dir = "{}{}/adv/adv_{}_{}".format(param.DATA_DIR, dataset_name, model_name, defense)

                for attack in attack_names :
                    # To-Do: to be modified after we generate using new paths.
                    if attack == param.BENIGN:
                        pass
                    else :
                        adv_path = "{}/{}/x_test.npy".format(defended_model_dir, attack)
                        if not os.path.exists(adv_path) :
                            raise ValueError(f"{adv_path} adv examples do not exist")
                
                
    
    ### RUN
    for dataset_name in model_dict.keys():

        ### load benign dataset
        x_train, y_train, x_test, y_test = load_data(dataset_name)

        for model_name in model_dict[dataset_name]:

            if metric == "accuracy" :
                baseline_accuracies= {}

            # model_defenses = {}
            
            '''load models for defense'''
            '''check dataset and adv examples'''
            for defense in defense_names :
                
                adv_dir = ""
                if defense == param.BENIGN:
                    
                    adv_dir = "{}{}/adv/{}".format(param.DATA_DIR, dataset_name, model_name)
                    
                    ### model path for benign model
                    model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
                    
                else :
                    adv_dir = "{}{}/adv/adv_{}_{}".format(param.DATA_DIR, dataset_name, model_name, defense)

                    ### model path for adversarially trained model
                    model_path = "{}{}/adv_{}_{}.h5".format(param.MODEL_DIR, dataset_name, model_name, defense)
                  
                    
                model_defense = load_model(model_path)

                if metric == "accuracy" :
                    baseline_model_path = "{}{}/{}.h5".format(param.MODEL_DIR, dataset_name, model_name)
                    baseline_model = load_model(model_path)

                    baseline_adv_dir = "{}{}/adv/{}".format(param.DATA_DIR, dataset_name, model_name)

                accuracies = []
                diffs = []
                mcr = [] # missclassification rates

                for attack in attack_names :
                    # To-Do: to be modified after we generate using new paths.
                    if attack == param.BENIGN:
                        adv_examples = x_test
                    else :
                        adv_path = "{}/{}/x_test.npy".format(adv_dir, attack)
                        adv_examples = np.load(adv_path)
                
                    criteria = AttackEvaluate(model_defense, x_test, y_test, adv_examples)
                    mr = criteria.misclassification_rate()
                    mr *= 100

                    mcr.append(mr)

                    if metric == "accuracy" :

                        accuracy = 100.0 - mr
                        accuracies.append(accuracy)
                        
                        if defense == param.BENIGN:
                            baseline_accuracies[attack] = accuracy
                            diffs.append(0.0)
                        else :
                            diffs.append(accuracy-baseline_accuracies[attack])
                
                row = f"& "
                # row = f"& {model_key} & "
                if metric == "missclassification_rate" :
                    for mc in mcr[:-1] :
                        row += f" {mc:.2f}\% &"
                    row += f" {mcr[-1]:.2f}\% \\\\"
                elif metric == "accuracy" :
                    for accuracy,  diff in zip(accuracies[:-1], diffs[:-1]) :
                        sign = "+" if diff >= 0 else "" 
                        row += f" {accuracy:.2f}\%({sign}{diff:.2f}\%) &"
                    
                    accuracy = accuracies[-1]
                    diff = diffs[-1]
                    sign = "+" if diff > 0 else "" 
                    row += f" {accuracy:.2f}\%({sign}{diff:.2f}\%) \\\\"
                else :
                    raise ValueError("Undefined metric")

                logger.info(row)
            
            
            
