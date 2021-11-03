import numpy as np
import tensorflow as tf
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    parser.add_argument("--model_layer", default=8, type=int)
    parser.add_argument("--order_number", default=0, type=int)

    args = parser.parse_args()

    model_name = args.model_name
    model_layer = args.model_layer
    dataset_name = args.dataset
    order_number = args.order_number

    # load dataset
    x_train, y_train, x_test, y_test = load_data(dataset_name)

    # import model
    from keras.models import load_model
    model_path = "{}{}/{}.h5".format(MODEL_DIR, dataset_name, model_name)
    model = load_model(model_path)
    model.summary()

    nc_index = {}
    nc_number = 0

    store_path = 'new_test/{}/{}'.format(dataset_name, model_name)
    os.makedirs(store_path, exist_ok=True)
    
    for order_number in range(2):
        for i in range(5000*order_number, 5000*(order_number+1)):
            new_image = mutate(x_test[i])
            if i == 5000*order_number+1000 or i == 5000*order_number+2000 or i == 5000*order_number+3000 or i == 5000*order_number+4000:
                print("-------------------------------------THIS IS {}-------------------------------------".format(i))
            if softmax(model.predict(np.expand_dims(new_image, axis=0))).argmax(axis=-1) != softmax(model.predict(np.expand_dims(x_test[i], axis=0))).argmax(axis=-1):
                nc_symbol = compare_nc(model, x_train, y_train, x_test, y_test, new_image, x_test[i], model_layer)
                if nc_symbol == True:
                    nc_index[i] = new_image
                    nc_number += 1

        print(nc_number)
        np.save(os.path.join(store_path, 'nc_index_test_{}.npy'.format(order_number)), nc_index)













