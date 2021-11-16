import tensorflow_datasets as tfds
import numpy as np
import os
import argparse

DATA_DIR = "../data/"

## convert array of indices to 1-hot encoded numpy array


def convert_label_to_one_hot_encoding(index_labels):
    one_hot_labels = np.zeros((index_labels.size, index_labels.max()+1))
    one_hot_labels[np.arange(index_labels.size), index_labels] = 1
    return one_hot_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='eurosat', type=str)
    args = parser.parse_args()

    # please check this link to browse more datasets
    # https://www.tensorflow.org/datasets/catalog/fashion_mnist
    # choices = ["fashion_mnist", "eurosat", "oxford_flowers102", "mnist", "cifar10"]
    dataset_name = args.dataset

    # batch_size=-1 to get the full dataset in NumPy arrays from the returned tf.Tensor object
    if dataset_name == "fashion_mnist" or dataset_name == "oxford_flowers102":
        dataset_train = tfds.load(
            name=dataset_name, split=tfds.Split.TRAIN, batch_size=-1)
        dataset_test = tfds.load(
            name=dataset_name, split=tfds.Split.TEST, batch_size=-1)
    elif dataset_name == "food101":
        dataset_train = tfds.load(
            name=dataset_name, split=tfds.Split.TRAIN, batch_size=-1)
        dataset_test = tfds.load(
            name=dataset_name, split=tfds.Split.VALIDATION, batch_size=-1)
    elif dataset_name == "eurosat":
        dataset_train = tfds.load(
            name=dataset_name, split="train[:80%]", batch_size=-1)
        dataset_test = tfds.load(
            name=dataset_name, split="train[80%:]", batch_size=-1)
    else:
        dataset_train = tfds.load(
            name=dataset_name, split=tfds.Split.TRAIN, batch_size=-1)
        dataset_test = tfds.load(
            name=dataset_name, split=tfds.Split.TEST, batch_size=-1)

    # tfds.as_numpy return a generator that yields NumPy array records out of a tf.data.Dataset
    dataset_train = tfds.as_numpy(dataset_train)
    dataset_test = tfds.as_numpy(dataset_test)

    # seperate the x and y
    x_train, y_train = dataset_train["image"], dataset_train["label"]
    x_test, y_test = dataset_test["image"], dataset_test["label"]

    y_train = convert_label_to_one_hot_encoding(y_train)
    y_test = convert_label_to_one_hot_encoding(y_test)

    os.makedirs(os.path.join(
        DATA_DIR, dataset_name + "/benign/"), exist_ok=True)
    np.save(os.path.join(DATA_DIR, dataset_name + "/benign/x_train.npy"), x_train)
    np.save(os.path.join(DATA_DIR, dataset_name + "/benign/y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, dataset_name + "/benign/x_test.npy"), x_test)
    np.save(os.path.join(DATA_DIR, dataset_name + "/benign/y_test.npy"), y_test)

    print("\n\nCHECK STANDARD\n")

    print(dataset_name)
    print(x_train.shape)
    print(y_train.shape)
    print(y_train[0])

    ## Check format/standard with the previous dataset
    from helper import load_data
    x_train, y_train, x_test, y_test = load_data("mnist")

    print("")
    print("mnist")
    print(x_train.shape)
    print(y_train.shape)
    print(y_train[0])

    x_train, y_train, x_test, y_test = load_data("cifar")

    print("")
    print("cifar")
    print(x_train.shape)
    print(y_train.shape)
    print(y_train[0])
