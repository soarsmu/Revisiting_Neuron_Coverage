import numpy as np
import tensorflow as tf
import argparse

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
    parser.add_argument("--model_layer", default=9, type=int)


    args = parser.parse_args()
    model_name = args.model_name
    model_layer = args.model_layer
    dataset_name = args.dataset
    # load dataset
    x_train, y_train, x_test, y_test = load_data(dataset_name)

    # import model
    from keras.models import load_model
    model = load_model('../data/' + dataset_name + '_data/model/' + model_name + '.h5')
    model.summary()

    order_number = 0

    nc_index = {}
    nc_number = 0
    for i in range(5000*order_number, 5000*(order_number+1)):
        new_image = mutate(x_train[i])

        if i % 1000 == 0:
            print("-------------------------------------THIS IS {}-------------------------------------".format(i))
        if softmax(model.predict(np.expand_dims(new_image, axis=0))).argmax(axis=-1) != softmax(model.predict(np.expand_dims(x_train[i], axis=0))).argmax(axis=-1):
            # find an adversarial example
            nc_symbol = compare_nc(model, x_train, y_train, x_test, y_test, new_image, x_train[i], model_layer)
            if nc_symbol == True:
                # new image can cover more neurons
                nc_index[i] = new_image
                nc_number += 1

    print(nc_number)
    np.save('fuzzing/nc_index_{}.npy'.format(order_number), nc_index)













