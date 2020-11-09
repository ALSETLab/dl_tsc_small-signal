import scipy.io as io
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import argparse

# Arguments to the function
parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--n_points", type = int)
parser.add_argument("--n_reduction", type = int)
args = parser.parse_args()

_dataset = args.dataset

# Validating number of points
if args.n_points:
    _n_points = args.n_points
    if _n_points > 505 or _n_points < 100:
        raise ValueError('Number of points not valid (tLeNet may not train correctly)')
else:
    _n_points = 505

if args.n_reduction:
    _reduce_dataset = True
    _n_reduction = int(np.abs(args.n_reduction))
    assert _n_reduction > 0, "Cannot reduce the training set by a zero proportion (division by zero)"
else:
    _reduce_dataset = False

if __name__ == "__main__":

    path_vr = os.path.abspath(os.path.join(os.getcwd(), 'train_data', _dataset, 'ResultsGenVr.mat'))
    path_vi = os.path.abspath(os.path.join(os.getcwd(), 'train_data', _dataset, 'ResultsGenVi.mat'))

    vr_mat = io.loadmat(path_vr)
    vr_mat = vr_mat['mts']
    vr_mat = vr_mat[0, 0]

    dt = vr_mat.dtype.names
    dt = list(dt)

    for i in range(len(dt)):
        if dt[i] == 'train':
            vr_train = vr_mat[i].reshape(max(vr_mat[i].shape))
        elif dt[i] == 'test':
            vr_test = vr_mat[i].reshape(max(vr_mat[i].shape))
        elif dt[i] == 'trainlabels':
            labels_train = vr_mat[i].reshape(max(vr_mat[i].shape))
        elif dt[i] == 'testlabels':
            labels_test = vr_mat[i].reshape(max(vr_mat[i].shape))

    vi_mat = io.loadmat(path_vi)
    vi_mat = vi_mat['mts']
    vi_mat = vi_mat[0, 0]

    dt = vi_mat.dtype.names
    dt = list(dt)

    for i in range(len(dt)):
        if dt[i] == 'train':
            vi_train = vi_mat[i].reshape(max(vi_mat[i].shape))
        elif dt[i] == 'test':
            vi_test = vi_mat[i].reshape(max(vi_mat[i].shape))
        elif dt[i] == 'trainlabels':
            labels_train = vi_mat[i].reshape(max(vi_mat[i].shape))
        elif dt[i] == 'testlabels':
            labels_test = vi_mat[i].reshape(max(vi_mat[i].shape))

    # Number of scenarios (i.e., length of the training/testing sets)
    n_train = vr_train.shape[0]
    n_test = vr_test.shape[0]
    n_scenarios = n_train + n_test

    # Number of points in each time-series
    length = _n_points
    print(f'Time-series length: {length}\n')

    # ---------------------------------------------------------
    # Training and validation sets
    # ---------------------------------------------------------

    x_train = np.zeros(shape = (n_train, length, 2))
    y_train = np.zeros(shape = (n_test, ))

    y_train = tf.keras.utils.to_categorical(labels_train - 1)

    for n in range(n_train):
        x_train[n, :, 0] = vr_train[n][1][0 : length]
        x_train[n, :, 1] = vi_train[n][1][0 : length]

    # Extracting validation dataset
    n_validation = int(0.1 * n_train)

    ind_validation = np.random.choice(np.arange(0, n_train), n_validation, replace = False)
    x_valid = x_train[ind_validation]
    y_valid = y_train[ind_validation]

    x_train = np.delete(x_train, ind_validation, axis = 0)
    y_train = np.delete(y_train, ind_validation, axis = 0)

    # Shuffling training set
    ind_train = np.arange(0, n_train - n_validation)
    np.random.shuffle(ind_train)
    x_train = x_train[ind_train]
    y_train = y_train[ind_train]

    # Shuffling validation set
    ind_validation = np.arange(0, n_validation)
    np.random.shuffle(ind_validation)
    x_valid = x_valid[ind_validation]
    y_valid = y_valid[ind_validation]

    # ---------------------------------------------------------
    # Testing set
    # ---------------------------------------------------------

    x_test = np.zeros(shape = (n_test, length, 2))
    y_test = np.zeros(shape = (n_test, ))

    y_test = tf.keras.utils.to_categorical(labels_test - 1)

    for n in range(n_train, n_scenarios):
        x_test[n - n_train, :, 0] = vr_test[n - n_train][1][0 : length]
        x_test[n - n_train, :, 1] = vi_test[n - n_train][1][0 : length]

    # Shuffling training set
    ind_test = np.arange(0, n_test)
    np.random.shuffle(ind_test)
    x_test = x_test[ind_test]
    y_test = y_test[ind_test]

    # ---------------------------------------------------------
    # All instances (for k-fold cross-validation)
    # ---------------------------------------------------------

    x_all = np.concatenate((x_train, x_test), axis = 0)
    y_all = np.concatenate((y_train, y_test), axis = 0)

    # Updating number of scenarios (removing validation set)
    n_scenarios -= n_validation

    # Shuffling all instances
    ind_all = np.arange(0, n_scenarios)
    np.random.shuffle(ind_all)
    x_all = x_all[ind_all]
    y_all = y_all[ind_all]

    print(f'x_train: {x_train.shape}')
    print(f'y_train: {y_train.shape}')
    print(f'x_valid: {x_valid.shape}')
    print(f'y_valid: {y_valid.shape}')
    print(f'x_test: {x_test.shape}')
    print(f'y_test: {y_test.shape}')
    print(f'x_all: {x_all.shape}')
    print(f'y_all: {y_all.shape}')

    path_save = os.path.join(os.getcwd(), 'train_data', f'{_dataset}_{_n_points}')
    if not os.path.exists(path_save):
        os.mkdir(path_save)

    np.save(os.path.join(path_save, 'x_all'), x_all, allow_pickle = False)
    np.save(os.path.join(path_save, 'x_train'), x_train, allow_pickle = False)
    np.save(os.path.join(path_save, 'x_test'), x_test, allow_pickle = False)
    np.save(os.path.join(path_save, 'x_valid'), x_valid, allow_pickle = False)

    np.save(os.path.join(path_save, 'y_all'), y_all, allow_pickle = False)
    np.save(os.path.join(path_save, 'y_train'), y_train, allow_pickle = False)
    np.save(os.path.join(path_save, 'y_test'), y_test, allow_pickle = False)
    np.save(os.path.join(path_save, 'y_valid'), y_valid, allow_pickle = False)

    data_all = {'x_train': x_train, 'x_test': x_test, 'x_all' : x_all, 'x_valid' : x_valid,
               'y_train': y_train, 'y_test': y_test, 'y_all' : y_all, 'y_valid' : y_valid}

    # Saving pickle file
    with open(os.path.join(path_save, 'data_all.pkl'), 'wb') as handle:
        pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if _reduce_dataset:
        # ----------------------------
        # Reduced Training Set
        # ----------------------------

        label_0 = np.where(np.argmax(y_train, axis = 1) == 0)[0]
        label_1 = np.where(np.argmax(y_train, axis = 1) == 1)[0]

        n0_train_red = int(label_0.shape[0] / _n_reduction)
        n1_train_red = int(label_1.shape[0] / _n_reduction)

        ind_0 = np.random.choice(label_0, size = n0_train_red, replace = False)
        ind_1 = np.random.choice(label_1, size = n1_train_red, replace = False)

        x_train_red = np.concatenate((x_train[ind_0], x_train[ind_1]), axis = 0)
        y_train_red = np.concatenate((y_train[ind_0], y_train[ind_1]), axis = 0)

        ind_train_red = np.arange(0, n0_train_red + n1_train_red)
        np.random.shuffle(ind_train_red)
        x_train_red = x_train_red[ind_train_red]
        y_train_red = y_train_red[ind_train_red]

        # ----------------------------
        # Test set
        # ----------------------------

        label_0 = np.where(np.argmax(y_test, axis = 1) == 0)[0]
        label_1 = np.where(np.argmax(y_test, axis = 1) == 1)[0]

        n0_test_red = int(label_0.shape[0] / _n_reduction)
        n1_test_red = int(label_1.shape[0] / _n_reduction)

        ind_0 = np.random.choice(label_0, size = n0_test_red, replace = False)
        ind_1 = np.random.choice(label_1, size = n1_test_red, replace = False)

        x_test_red = np.concatenate((x_test[ind_0], x_test[ind_1]), axis = 0)
        y_test_red = np.concatenate((y_test[ind_0], y_test[ind_1]), axis = 0)

        # -------------------------------------
        # All set (for k-fold cross-validation)
        # -------------------------------------

        x_all_red = np.concatenate((x_train_red, x_test_red), axis = 0)
        y_all_red = np.concatenate((y_train_red, y_test_red), axis = 0)

        ind_all_red = np.arange(0, n0_train_red + n1_train_red + n0_test_red + n1_test_red)
        np.random.shuffle(ind_all_red)
        x_all_red = x_all_red[ind_all_red]
        y_all_red = y_all_red[ind_all_red]

        x_valid_red = x_valid
        y_valid_red = y_valid

        print(f'\nReduced Dataset\n')
        print(f'x_train_red: {x_train_red.shape}')
        print(f'y_train_red: {y_train_red.shape}')
        print(f'x_valid_red: {x_valid_red.shape}')
        print(f'y_valid_red: {y_valid_red.shape}')
        print(f'x_test_red (not saved; inside x_all_red): {x_test_red.shape}')
        print(f'y_test_red (not saved; inside y_all_red): {y_test_red.shape}')
        print(f'x_all_red: {x_all_red.shape}')
        print(f'y_all_red: {y_all_red.shape}')

        path_save = os.path.join(os.getcwd(), 'train_data', f'{_dataset}_{_n_reduction}red_{_n_points}')
        if not os.path.exists(path_save):
            os.mkdir(path_save)

        np.save(os.path.join(path_save, 'x_all'), x_all_red, allow_pickle = False)
        np.save(os.path.join(path_save, 'x_train'), x_train_red, allow_pickle = False)
        np.save(os.path.join(path_save, 'x_test'), x_test, allow_pickle = False)
        np.save(os.path.join(path_save, 'x_valid'), x_valid_red, allow_pickle = False)

        np.save(os.path.join(path_save, 'y_all'), y_all, allow_pickle = False)
        np.save(os.path.join(path_save, 'y_train'), y_train_red, allow_pickle = False)
        np.save(os.path.join(path_save, 'y_test'), y_test, allow_pickle = False)
        np.save(os.path.join(path_save, 'y_valid'), y_valid_red, allow_pickle = False)

        data_all = {'x_train': x_train_red, 'x_test': x_test, 'x_all' : x_all_red, 'x_valid' : x_valid_red,
                   'y_train': y_train_red, 'y_test': y_test, 'y_all' : y_all_red, 'y_valid' : y_valid_red}

        # Saving pickle file
        with open(os.path.join(path_save, 'data_all.pkl'), 'wb') as handle:
            pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
