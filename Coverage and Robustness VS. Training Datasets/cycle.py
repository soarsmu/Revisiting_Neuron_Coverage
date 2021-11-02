

from helper import load_data, mutate, softmax, compare_nc, retrain
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
        print(i)
        if i == 10:
            print('.', end='')
            break
        if softmax(current_model.predict(np.expand_dims(new_image, axis=0))).argmax(axis=-1) != softmax(current_model.predict(np.expand_dims(x_train[i], axis=0))).argmax(axis=-1):
            # find an adversarial example
            nc_symbol = compare_nc(current_model, x_train, y_train, x_test, y_test, new_image, x_train[i], model_layer)
            if nc_symbol == True:
                # new image can cover more neurons
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
    retrained_model = retrain(current_model, x_train, y_train, x_test, y_test, batch_size=32, epochs=5)
    new_model_path = "{}{}/{}/{}.h5".format(THIS_MODEL_DIR, dataset_name, model_name, str(T))
    retrained_model.save(new_model_path)

    print("Done")
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='lenet1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--model_layer", default=8, type=int)
    parser.add_argument("--order_number", default=0, type=int)
    parser.add_argument("--improve_coverage", default=True, type=bool)

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
    
    # Save it under this folder
    new_model_path = "{}{}/{}/{}/{}.h5".format(THIS_MODEL_DIR, dataset_name, model_name, is_improve, str(0))
    model.save(new_model_path)

    cycle(1)