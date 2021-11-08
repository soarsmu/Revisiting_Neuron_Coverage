import numpy as np
import tensorflow as tf
import os
import argparse

from tqdm import tqdm

MODEL_DIR = "../models/"

####for solving some specific problems, don't care
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import warnings
from helper import load_data, softmax, compare_nc, mutate

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='lenet1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)

    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset

    # load dataset
    x_train, y_train, x_test, y_test = load_data(dataset_name)

    # import model
    from keras.models import load_model
    model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset_name, model_name)
    model = load_model(model_path)
    model_layer = len(model.layers) - 1

    store_path = 'new_test/{}/{}'.format(dataset_name, model_name)
    os.makedirs(store_path, exist_ok=True)
    new_images = []
    for i in tqdm(range(len(x_test)), desc="transformation ......"):
        new_images.append(mutate(x_test[i]))

    for order_number in range(2):
        nc_index = {}
        nc_number = 0
        for i in tqdm(range(5000*order_number, 5000*(order_number+1), 500), desc="Total progress:"):
            for index, (pred_new, pred_old) in enumerate(zip(softmax(model.predict(np.array(new_images[i:i+500]))).argmax(axis=-1), softmax(model.predict(x_test[i:i+500])).argmax(axis=-1))):
                nc_symbol = compare_nc(model, x_train, y_train, x_test, y_test, new_images[i+index], x_test[i+index], model_layer)
                if nc_symbol == True:
                    nc_index[i+index] = new_images[i+index]
                    nc_number += 1

        print("Log: new image can cover more neurons: {}".format(nc_number))
        np.save(os.path.join(store_path, 'nc_index_test_{}.npy'.format(order_number)), nc_index)

    for order_number in range(2):
        index = np.load(os.path.join(store_path, 'nc_index_test_{}.npy'.format(order_number)), allow_pickle=True).item()
        for y, x in index.items():
            # print(y)
            x_test[y] = x

    np.save(os.path.join(store_path, 'x_test_new.npy'), x_test)













