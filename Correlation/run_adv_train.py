import multiprocessing
from multiprocessing import Pool, Process, Queue, Manager
import os
from adv_train_example import check_data_path
import tensorflow as tf




def train(gpu_id):
    while True:
        if not q.empty():
            attack_name, dataset, model_name = q.get()
            cmd = 'CUDA_VISIBLE_DEVICES=' + gpu_id + ' python adv_train_example.py -dataset ' + dataset + ' -model ' + model_name + ' -attack ' + attack_name
            os.system(cmd)
        else:
            print("Finished")
            return

if __name__=='__main__':

    DATA_DIR = "../data/"
    MODEL_DIR = "../models/"

    datasets = ['cifar', 'mnist', 'svhn']
    model_dict = {
                'mnist': ['lenet1', 'lenet4', 'lenet5'],
                'cifar': ['vgg16', 'resnet20'],
                'svhn' : ['svhn_model', 'svhn_second', 'svhn_first']
                }

    attack_names = ['fgsm', 'pgd', 'bim']



    ### verify path
    for dataset_name in model_dict.keys():
        # verify data path
        check_data_path(dataset_name)
        # verify model path
        for model_name in model_dict[dataset_name]:
            model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset_name, model_name)
            assert os.path.exists(model_path)




    ### add combinations into queues
    manager = multiprocessing.Manager()
    q = manager.Queue()

    for attack_name in attack_names:
        for dataset in datasets:
            for model_name in model_dict[dataset]:
                q.put((attack_name, dataset, model_name))




    p_list = []
    for i in range(tf.contrib.eager.num_gpus()):
        gpu_id = i
        p = multiprocessing.Process(target=train, args=(str(gpu_id), ))
        p_list.append(p)
        p.start()
    
    for i in p_list:
        i.join()

    print("All processed finished.")
