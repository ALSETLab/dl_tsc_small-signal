import os
import numpy as np
import random

def load_dataset(dataset, _validation = 'normal', working_directory = None):

    if working_directory == None:
        working_directory = os.getcwd()
        print("\nload_dataset: it is assumed that `train_data` is in the current working directory\n")

    path_data = os.path.abspath(os.path.join(working_directory, "train_data", dataset))

    x_all_path = os.path.join(path_data, "x_all.npy")
    x_train_path = os.path.join(path_data, "x_train.npy")
    x_test_path = os.path.join(path_data, "x_test.npy")
    x_valid_path = os.path.join(path_data, "x_valid.npy")

    y_all_path = os.path.join(path_data, "y_all.npy")
    y_train_path = os.path.join(path_data, "y_train.npy")
    y_test_path = os.path.join(path_data, "y_test.npy")
    y_valid_path = os.path.join(path_data, "y_valid.npy")

    x_all = np.load(x_all_path, allow_pickle = False)
    x_train = np.load(x_train_path, allow_pickle = False)
    x_test = np.load(x_test_path, allow_pickle = False)
    x_valid = np.load(x_valid_path, allow_pickle = False)

    y_all = np.load(y_all_path, allow_pickle = False)
    y_train = np.load(y_train_path, allow_pickle = False)
    y_test = np.load(y_test_path, allow_pickle = False)
    y_valid = np.load(y_valid_path, allow_pickle = False)

    input_shape = (x_all.shape[1], x_all.shape[2])
    nb_classes = y_all.shape[-1]

    return {'x_all' : x_all, 'x_train' : x_train, 'x_test' : x_test, 'x_valid' : x_valid,
            'y_all' : y_all, 'y_train' : y_train, 'y_test' : y_test, 'y_valid' : y_valid}, input_shape, nb_classes
