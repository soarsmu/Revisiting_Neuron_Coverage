## 
# Source: https://github.com/jerett/Keras-CIFAR10/blob/master/small_resnet56.ipynb
import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from resnet import ResNet20, ResNet32, ResNet56
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit


class Solver(object):
    """
    A Solver encapsulates all the logic nessary for training cifar10 classifiers.The train model is defined
    outside, you must pass it to init().
    The solver train the model, plot loss and aac history, and test on the test data.
    Example usage might look something like this.
    model = MyAwesomeModel(opt=SGD, losses='categorical_crossentropy',  metrics=['acc'])
    model.compile(...)
    model.summary()
    solver = Solver(model)
    history = solver.train()
    plotHistory(history)
    solver.test()
    """

    def __init__(self, model, data):
        """
        :param model: A model object conforming to the API described above
        :param data:  A tuple of training, validation and test data from CIFAR10Data
        """
        self.model = model
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = data

    def train(self, epochs=200, batch_size=128, data_augmentation=True, callbacks=None):
        if data_augmentation:
            # datagen
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                # rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=4,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=4,
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
            )
            # (std, mean, and principal components if ZCA whitening is applied).
            # datagen.fit(x_train)
            print('train with data augmentation')
            train_gen = datagen.flow(
                self.X_train, self.Y_train, batch_size=batch_size)
            history = self.model.fit_generator(generator=train_gen,
                                               epochs=epochs,
                                               callbacks=callbacks,
                                               validation_data=(
                                                   self.X_val, self.Y_val),
                                               )
        else:
            print('train without data augmentation')
            history = self.model.fit(self.X_train, self.Y_train,
                                     batch_size=batch_size, epochs=epochs,
                                     callbacks=callbacks,
                                     validation_data=(self.X_val, self.Y_val),
                                     )
        return history

    def test(self):
        loss, acc = self.model.evaluate(self.X_test, self.Y_test)
        print('test data loss:%.2f acc:%.4f' % (loss, acc))


def lr_scheduler(epoch):
    new_lr = lr
    if epoch <= 91:
        pass
    elif epoch > 91 and epoch <= 137:
        new_lr = lr * 0.1
    else:
        new_lr = lr * 0.01
    print('new lr:%.2e' % new_lr)
    return new_lr

def convert_label_to_one_hot_encoding(index_labels):
    one_hot_labels = np.zeros((index_labels.size, index_labels.max()+1))
    one_hot_labels[np.arange(index_labels.size), index_labels] = 1
    return one_hot_labels


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='resnet56', type=str)
    parser.add_argument("--dataset", default='cifar10', type=str)
    parser.add_argument("--num_classes", default=10, type=int)

    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset
    num_classes = args.num_classes

    if dataset_name == "cifar10" or dataset_name == "cifar100":
        
        train_ds = tfds.load(name=dataset_name, split=tfds.Split.TRAIN, batch_size=-1)
        test_ds = tfds.load(name=dataset_name, split=tfds.Split.TEST, batch_size=-1)

        # tfds.as_numpy return a generator that yields NumPy array records out of a tf.data.Dataset
        dataset_train = tfds.as_numpy(train_ds)
        dataset_test = tfds.as_numpy(test_ds)

        # seperate the x and y
        x_train, y_train = dataset_train["image"], dataset_train["label"]
        x_test, y_test = dataset_test["image"], dataset_test["label"]

        y_train = convert_label_to_one_hot_encoding(y_train)
        y_test = convert_label_to_one_hot_encoding(y_test)

        num_train = int(x_train.shape[0] * 0.9)
        num_val = x_train.shape[0] - num_train
        mask = list(range(num_train, num_train+num_val))
        x_val = x_train[mask]
        y_val = y_train[mask]

        mask = list(range(num_train))
        x_train = x_train[mask]
        y_train = y_train[mask]
    
    elif  dataset_name == "eurosat" :
        train_ds = tfds.load(name=dataset_name, split=tfds.Split.TRAIN, batch_size=-1)
        
        dataset_train = tfds.as_numpy(train_ds)
        x_train, y_train = dataset_train["image"], dataset_train["label"]
        y_train = convert_label_to_one_hot_encoding(y_train)
        
        num_train = int(x_train.shape[0] * 0.8)
        num_val = int(x_train.shape[0] * 0.1)
        num_test = x_train.shape[0] - num_train - num_val
        
        mask = list(range(num_train))
        x_train = x_train[mask]
        y_train = y_train[mask]

        mask = list(range(num_train, num_train+num_val))
        x_val = x_train[mask]
        y_val = y_train[mask]

        mask = list(range(num_train+num_val, num_train+num_val+num_test))
        x_test = x_train[mask]
        y_test = y_train[mask]


    data = (x_train, y_train, x_val, y_val, x_test, y_test)

    
    weight_decay = 1e-4
    lr = 1e-1

    if dataset_name == "cifar10" or dataset_name == "cifar100":
        input_shape = (32, 32, 3)
    elif dataset_name == "eurosat":
        input_shape = (64, 64, 3)
    
    if model_name == "resnet20" :
        model = ResNet20(input_shape=input_shape, classes=num_classes, weight_decay=weight_decay)
    elif model_name == "resnet32":
        model = ResNet32(input_shape=input_shape, classes=num_classes, weight_decay=weight_decay)
    elif model_name == "resnet56":
        model = ResNet56(input_shape=input_shape, classes=num_classes, weight_decay=weight_decay)
    else:
        raise ValueError("Undefined model")

    opt = optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
    model.compile(optimizer=opt,
                    loss=losses.categorical_crossentropy,
                    metrics=['accuracy'])
    model.summary()


    reduce_lr = LearningRateScheduler(lr_scheduler)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    
    MODEL_DIR = "../models/"
    os.makedirs(MODEL_DIR + dataset_name, exist_ok=True)

    best_model_name = '{}{}/{}_best_model.h5'.format(
        MODEL_DIR, dataset_name, model_name)
    mc = ModelCheckpoint(best_model_name, monitor='val_loss',
                         mode='min', save_best_only=True)

    solver = Solver(model, data)
    history = solver.train(epochs=182, batch_size=128,
                        data_augmentation=True, callbacks=[reduce_lr, mc, es])
    solver.test()

    model.save("{}{}/{}.h5".format(MODEL_DIR, dataset_name, model_name))


    
