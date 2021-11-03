import numpy as np
from helper import load_data
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='lenet1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--model_layer", default=8, type=int)
    parser.add_argument("--order_number", default=0, type=int)

    args = parser.parse_args()

    model_name = args.model_name

    dataset_name = args.dataset
    order_number = args.order_number
    x_train, y_train, x_test, y_test = load_data(dataset_name)

    store_path = 'new_test/{}/{}'.format(dataset_name, model_name)
    assert os.path.exists(store_path)

    for order_number in range(2):
        index = np.load(os.path.join(store_path, 'nc_index_test_{}.npy'.format(order_number)), allow_pickle=True).item()
        for y, x in index.items():
            print(y)
            x_test[y] = x

    np.save(os.path.join(store_path, 'x_test_new.npy'), x_test)
