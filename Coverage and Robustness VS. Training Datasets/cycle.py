

from helper import load_data, mutate, softmax, compare_nc, retrain
from helper import Coverage
from helper import AttackEvaluate
import argparse
import tensorflow as tf
import numpy as np
from keras.models import load_model
import os


DATA_DIR = "../data/"
MODEL_DIR = "../models/"
THIS_MODEL_DIR = "./models/"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

def evaluate_robustness(T, retrained_model, x_test, y_test, x_test_new):
    path_to_result = "robustness_results/{}/{}/{}".format(dataset_name, model_name, is_improve)
    os.makedirs(path_to_result, exist_ok=True)

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

    with open(os.path.join(path_to_result, "robustness_metrics_{}.txt".format(T)), "a") as f:
        f.write("\n------------------------------------------------------------------------------\n")
        f.write('the result of {} {} is: \n'.format(dataset_name, model_name))
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


def evaluate_coverage(model, l, T, x_train, y_train, x_test, y_test):
    path_to_result = "coverage_results/{}/{}/{}".format(dataset_name, model_name, is_improve)
    os.makedirs(path_to_result, exist_ok=True)

    coverage = Coverage(model, x_train, y_train, x_test, y_test, x_train)
    # Based on the original code to evaluate Coverage, x_train and x_train_adv is the same.
    # In the Coverage class, only the last parameter is used.
    nc1, activate_num1, total_num1 = coverage.NC(l, threshold=0.1)
    nc2, activate_num2, total_num2 = coverage.NC(l, threshold=0.3)
    nc3, activate_num3, total_num3 = coverage.NC(l, threshold=0.5)
    nc4, activate_num4, total_num4 = coverage.NC(l, threshold=0.7)
    nc5, activate_num5, total_num5 = coverage.NC(l, threshold=0.9)
    tknc, pattern_num, total_num6 = coverage.TKNC(l)
    tknp = coverage.TKNP(l)
    kmnc, nbc, snac, covered_num, l_covered_num, u_covered_num, neuron_num = coverage.KMNC(l)

    with open(os.path.join(path_to_result, "coverage_result_{}.txt".format(T)), "a") as f:
        f.write("\n------------------------------------------------------------------------------\n")
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


def cycle(T: int):
    assert T > 0

    # Step 1. Load the current model M_i
    current_model_path = "{}{}/{}/{}/{}.h5".format(THIS_MODEL_DIR, dataset_name, model_name, is_improve, str(T - 1))
    current_model = load_model(current_model_path)

    # Step 2. According to the current M_i and dataset, generate examples T_i
    ## Load the current dataset we have
    x_train, y_train, x_test, y_test = load_data(dataset_name)
    for i in range(T-1):
        index = np.load('fuzzing/nc_index_{}.npy'.format(i), allow_pickle=True).item()
        for y, x in index.items():
            x_train = np.concatenate((x_train, np.expand_dims(x, axis=0)), axis=0)
            y_train = np.concatenate((y_train, np.expand_dims(y_train[y], axis=0)), axis=0)

    ## Generate new examples
    nc_index = {}
    nc_number = 0
    for i in range(5000*(T-1), 5000*(T)):
        new_image = mutate(x_train[i])
        if i % 100 == 0:
            print('.', end='')
            break
        if softmax(current_model.predict(np.expand_dims(new_image, axis=0))).argmax(axis=-1) != softmax(current_model.predict(np.expand_dims(x_train[i], axis=0))).argmax(axis=-1):
            # find an adversarial example
            nc_symbol = compare_nc(current_model, x_train, y_train, x_test, y_test, new_image, x_train[i], model_layer)
            if nc_symbol and improve_coverage:
                # new image can cover more neurons, and we want such improvements
                    nc_index[i] = new_image
                    nc_number += 1
            
            if (not improve_coverage) and (not nc_symbol):
                # new image CANNOT cover more neurons, and we want examples cannot improve coverage
                    nc_index[i] = new_image
                    nc_number += 1


    print(nc_number)
    data_folder = 'fuzzing/{}/{}/{}'.format(dataset_name, model_name, is_improve)
    os.makedirs(data_folder, exist_ok=True)
    np.save(os.path.join(data_folder, 'nc_index_{}.npy'.format(T)), nc_index)

    # Step 3. Retrain M_i against T_i, to obtain M_{i+1}
    ## Augment the newly generate examples into the training data

    index = np.load(os.path.join(data_folder, 'nc_index_{}.npy'.format(T)), allow_pickle=True).item()
    for y, x in index.items():
        x_train = np.concatenate((x_train, np.expand_dims(x, axis=0)), axis=0)
        y_train = np.concatenate((y_train, np.expand_dims(y_train[y], axis=0)), axis=0)

    ## Retrain the model
    retrained_model = retrain(current_model, x_train, y_train, x_test, y_test, batch_size=128, epochs=5)
    new_model_path = "{}{}/{}/{}/{}.h5".format(THIS_MODEL_DIR, dataset_name, model_name, is_improve, str(T))
    retrained_model.save(new_model_path)


    # Step 4. Evaluate the current model
    ## Evaluate coverage
    evaluate_coverage(retrained_model, l, T, x_train, y_train, x_test, y_test)

    ## Evaluate robustness
    x_test_new = np.load('x_test_new.npy') # To-Do: Need to be fixed! This is only for mnist + lenet1.
    evaluate_robustness(T, retrained_model, x_test, y_test, x_test_new)

    print("Done")
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='lenet1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--model_layer", default=8, type=int)
    parser.add_argument("--order_number", default=0, type=int)
    parser.add_argument("--improve_coverage", action='store_true')

    args = parser.parse_args()
    model_name = args.model_name
    model_layer = args.model_layer
    dataset_name = args.dataset
    order_number = args.order_number
    improve_coverage = args.improve_coverage


    is_improve = 'improve' if improve_coverage else 'no_improve'

    # Load the original model
    model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset_name, model_name)
    model = load_model(model_path)
    model_layer = len(model.layers) - 1
    l = [0, model_layer]
    
    # Save it under this folder
    new_model_path = "{}{}/{}/{}/{}.h5".format(THIS_MODEL_DIR, dataset_name, model_name, is_improve, str(0))
    model.save(new_model_path)

    for order_number in range(1, 11):
        cycle(order_number)