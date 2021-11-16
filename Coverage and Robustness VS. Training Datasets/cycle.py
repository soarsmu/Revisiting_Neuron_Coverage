

from helper import load_data, mutate, softmax, compare_nc, retrain
from helper import Coverage
from helper import AttackEvaluate
import argparse
import tensorflow as tf
import numpy as np
from keras.models import load_model
import os
from tqdm import tqdm


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

    with open(os.path.join(path_to_result, "robustness_metrics_{}.txt".format(T)), "w") as f:
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
    nc, activate_num, total_num = coverage.NC(l, threshold=[0.1, 0.3, 0.5, 0.7, 0.9])
    # nc2, activate_num2, total_num2 = coverage.NC(l, threshold=0.3)
    # nc3, activate_num3, total_num3 = coverage.NC(l, threshold=0.5)
    # nc4, activate_num4, total_num4 = coverage.NC(l, threshold=0.7)
    # nc5, activate_num5, total_num5 = coverage.NC(l, threshold=0.9)
    for threshold in nc.keys():
        print('NC({}): {}  activate_num: {}  total_num: {}'.format(threshold, nc[threshold], activate_num[threshold], total_num))

    tknc, pattern_num, total_num = coverage.TKNC(l)
    print('TKNC: {}  pattern_num: {}  total_num: {}'.format(tknc, pattern_num, total_num))
    
    tknp = coverage.TKNP(l)
    print('TKNP: {}'.format(tknp))
        
    kmnc, nbc, snac, covered_num, l_covered_num, u_covered_num, neuron_num = coverage.KMNC(l)
    print('KMNC: {}  covered_num: {}  total_num: {}'.format(kmnc, covered_num, neuron_num))
    print('NBC: {}  l_covered_num: {}  u_covered_num: {}'.format(nbc, l_covered_num, u_covered_num))
    print('SNAC: {}'.format(snac))

    with open(os.path.join(path_to_result, "coverage_result_{}.txt".format(T)), "w") as f:
        f.write("\n------------------------------------------------------------------------------\n")
        f.write('the coverage of {} {} is: \n'.format(dataset_name, model_name))
        for threshold in nc.keys():
            f.write('NC({}): {}  activate_num: {}  total_num: {} \n'.format(threshold, nc[threshold], activate_num[threshold], total_num))
        # f.write('NC(0.3): {}  activate_num: {}  total_num: {} \n'.format(nc[0.3], activate_num[0.3], total_num))
        # f.write('NC(0.5): {}  activate_num: {}  total_num: {} \n'.format(nc[0.5], activate_num[0.5], total_num))
        # f.write('NC(0.7): {}  activate_num: {}  total_num: {} \n'.format(nc[0.7], activate_num[0.7], total_num))
        # f.write('NC(0.9): {}  activate_num: {}  total_num: {} \n'.format(nc[0.9], activate_num[0.9], total_num))
        f.write('TKNC: {}  pattern_num: {}  total_num: {} \n'.format(tknc, pattern_num, total_num))
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

    if not os.path.exists(os.path.join('new_test/{}/{}'.format(dataset_name, model_name), 'x_test_new.npy')):
        print("Generate test set")
        
        new_images = []
        for i in tqdm(range(len(x_test)), desc="transformation ......"):
            new_images.append(mutate(x_test[i]))

        nc_index = {}
        nc_number = 0
        for i in tqdm(range(0, len(x_test), 500), desc="Total progress:"):
            for index, (pred_new, pred_old) in enumerate(zip(softmax(model.predict(np.array(new_images[i:i+500]))).argmax(axis=-1), softmax(model.predict(x_test[i:i+500])).argmax(axis=-1))):
                nc_symbol = compare_nc(model, x_train, y_train, x_test, y_test, new_images[i+index], x_test[i+index], model_layer)
                if nc_symbol == True:
                    nc_index[i+index] = new_images[i+index]
                    nc_number += 1
        
        print("Log: new image can cover more neurons: {}".format(nc_number))
        store_path = 'new_test/{}/{}'.format(dataset_name, model_name)
        os.makedirs(store_path, exist_ok=True)
        for y, x in nc_index.items():
            x_test[y] = x
        np.save(os.path.join(store_path, 'x_test_new.npy'), x_test)

    data_folder = 'fuzzing/{}/{}/{}'.format(dataset_name, model_name, is_improve)
    os.makedirs(data_folder, exist_ok=True)
    
    if not os.path.exists(os.path.join(data_folder, "new_images.npy")):
        print("Log: Start do transformation in images")
        new_images = []
        for i in tqdm(range(len(x_train))):
            new_images.append(mutate(x_train[i]))
        np.save(os.path.join(data_folder, "new_images.npy"), new_images)
    else:
        print("Log: Load mutantions.")
        new_images = np.load(os.path.join(data_folder, "new_images.npy"))

    for i in range(1, T):
        index = np.load('fuzzing/{}/{}/{}/nc_index_{}.npy'.format(dataset_name, model_name, is_improve, i), allow_pickle=True).item()
        for y, x in index.items():
            x_train = np.concatenate((x_train, np.expand_dims(x, axis=0)), axis=0)
            y_train = np.concatenate((y_train, np.expand_dims(y_train[y], axis=0)), axis=0)

    if not os.path.exists(os.path.join(data_folder, 'nc_index_{}.npy'.format(T))):
        ## Generate new examples
        nc_index = {}
        nc_number = 0
        for i in tqdm(range(5000*(T-1), 5000*(T), 500), desc="Total progress:"):
            for index, (pred_new, pred_old) in enumerate(zip(softmax(current_model.predict(np.array(new_images[i:i+500]))).argmax(axis=-1), softmax(current_model.predict(x_train[i:i+500])).argmax(axis=-1))):
                # find an adversarial example
                if pred_new != pred_old:
                    nc_symbol = compare_nc(current_model, x_train, y_train, x_test, y_test, new_images[i+index], x_train[i+index], model_layer)
                    if nc_symbol and improve_coverage:
                        # new image can cover more neurons, and we want such improvements
                            nc_index[i+index] = new_images[i+index]
                            nc_number += 1
                    
                    if (not improve_coverage) and (not nc_symbol):
                        # new image CANNOT cover more neurons, and we want examples cannot improve coverage
                            nc_index[i+index] = new_images[i+index]
                            nc_number += 1


        print("Log: new image can/cannot cover more neurons: {}".format(nc_number))
        
        np.save(os.path.join(data_folder, 'nc_index_{}.npy'.format(T)), nc_index)

    # Step 3. Retrain M_i against T_i, to obtain M_{i+1}
    ## Augment the newly generate examples into the training data

    index = np.load(os.path.join(data_folder, 'nc_index_{}.npy'.format(T)), allow_pickle=True).item()
    for y, x in index.items():
        x_train = np.concatenate((x_train, np.expand_dims(x, axis=0)), axis=0)
        y_train = np.concatenate((y_train, np.expand_dims(y_train[y], axis=0)), axis=0)


    # Step 4. Evaluate the current model
    ## Evaluate coverage
    print(x_train.shape)
    print("\nEvaluate coverage ......")
    evaluate_coverage(current_model, l, T, x_train, y_train, x_test, y_test)

    ## Evaluate robustness
    print("\nEvaluate robustness ......")
    store_path = 'new_test/{}/{}'.format(dataset_name, model_name)
    x_test_new = np.load(os.path.join(store_path, 'x_test_new.npy'),  allow_pickle=True)
    evaluate_robustness(T, current_model, x_test, y_test, x_test_new)

    ## Retrain the model
    retrained_model = retrain(current_model, x_train, y_train, x_test, y_test, batch_size=128, epochs=5)
    new_model_path = "{}{}/{}/{}/{}.h5".format(THIS_MODEL_DIR, dataset_name, model_name, is_improve, str(T))
    retrained_model.save(new_model_path)

    print("Done\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='lenet1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--improve_coverage", action='store_true')

    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset
    improve_coverage = args.improve_coverage

    is_improve = 'improve' if improve_coverage else 'no_improve'

    # Load the original model
    model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset_name, model_name)
    model = load_model(model_path)
    model_layer = len(model.layers)
    l = range(model_layer)

    # Save it under this folder
    new_model_path = "{}{}/{}/{}/{}.h5".format(THIS_MODEL_DIR, dataset_name, model_name, is_improve, str(0))
    model.save(new_model_path)

    for order_number in range(1, 11):
        cycle(order_number)