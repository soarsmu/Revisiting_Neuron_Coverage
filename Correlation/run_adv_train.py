import sys
sys.path.append('..')
import parameters as param


from utils import get_available_gpus
import multiprocessing
from multiprocessing import Pool, Process, Queue, Manager
import os
import tensorflow as tf




def train(gpu_id):
    while True:
        if not q.empty():
            attack_name, dataset, model_name = q.get()
            cmd = 'CUDA_VISIBLE_DEVICES=' + gpu_id + ' python adv_train.py --dataset ' + dataset + ' --model ' + model_name + ' --attack ' + attack_name + ' --nb_epochs 200 --batch_size 256'
            os.system(cmd)
        else:
            print("Finished")
            return

if __name__=='__main__':


    # datasets = ['cifar', 'mnist', 'svhn', 'eurosat']
    # model_dict = {
    #             'mnist': ['lenet1', 'lenet4', 'lenet5'],
    #             'cifar': ['vgg16', 'resnet20', 'resnet56'],
    #             'svhn' : ['svhn_model', 'svhn_second', 'svhn_first'],
    #             'eurosat': ['resnet20', 'resnet56']
    #             }

    datasets = ['cifar', 'mnist', 'svhn', 'eurosat']
    model_dict = {
                'mnist': ['lenet5'],
                'cifar': ['vgg16'],
                'svhn' : ['svhn_second'],
                'eurosat': ['resnet56']
                }


    attack_names = ['fgsm', 'pgd', 'deepfool', 'bim', 'apgd','cw']

    ### add combinations into queues
    manager = multiprocessing.Manager()
    q = manager.Queue()

    for attack_name in attack_names:
        for dataset in datasets:
            for model_name in model_dict[dataset]:
                q.put((attack_name, dataset, model_name))




    p_list = []

    for i in range(len(get_available_gpus())):
        gpu_id = i
        p = multiprocessing.Process(target=train, args=(str(gpu_id), ))
        p_list.append(p)
        p.start()
    
    for i in p_list:
        i.join()

    print("All processed finished.")
