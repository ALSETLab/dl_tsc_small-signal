import pickle
import numpy as np
import argparse

# Arguments to the function
parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--reduced", type = bool)
args = parser.parse_args()

_dataset = args.dataset

if args.reduced:
    _reduced = args.reduced
else:
    _reduced = False

if __name__ == "__main__":

    with open(f'train_data/{_dataset}/data_all.pkl', 'rb') as handle:
        data_all = pickle.load(handle)

    x_train = data_all['x_train']
    print(f'x_train: {x_train.shape}')
    x_test = data_all['x_test']
    print(f'x_test: {x_test.shape}')
    x_all = data_all['x_all']
    print(f'x_all: {x_all.shape}')
    x_valid = data_all['x_valid']
    print(f'x_valid: {x_valid.shape}')

    y_train = data_all['y_train']
    print(f'y_train: {y_train.shape}')
    y_test = data_all['y_test']
    print(f'y_test: {y_test.shape}')
    y_all = data_all['y_all']
    print(f'y_all: {y_all.shape}')
    y_valid = data_all['y_valid']
    print(f'y_valid: {y_valid.shape}')

    np.save(f'train_data/{_dataset}/x_all', x_all, allow_pickle = False)
    np.save(f'train_data/{_dataset}/x_train', x_train, allow_pickle = False)
    np.save(f'train_data/{_dataset}/x_test', x_test, allow_pickle = False)
    np.save(f'train_data/{_dataset}/x_valid', x_valid, allow_pickle = False)

    np.save(f'train_data/{_dataset}/y_all', y_all, allow_pickle = False)
    np.save(f'train_data/{_dataset}/y_train', y_train, allow_pickle = False)
    np.save(f'train_data/{_dataset}/y_test', y_test, allow_pickle = False)
    np.save(f'train_data/{_dataset}/y_valid', y_valid, allow_pickle = False)

    if _reduced:

        print('\nReduced dataset \n')

        with open(f'train_data/{_dataset}_red/data_all.pkl', 'rb') as handle:
            data_all = pickle.load(handle)

        x_train = data_all['x_train']
        print(f'x_train: {x_train.shape}')
        x_test = data_all['x_test']
        print(f'x_test: {x_test.shape}')
        x_all = data_all['x_all']
        print(f'x_all: {x_all.shape}')
        x_valid = data_all['x_valid']
        print(f'x_valid: {x_valid.shape}')

        y_train = data_all['y_train']
        print(f'y_train: {y_train.shape}')
        y_test = data_all['y_test']
        print(f'y_test: {y_test.shape}')
        y_all = data_all['y_all']
        print(f'y_all: {y_all.shape}')
        y_valid = data_all['y_valid']
        print(f'y_valid: {y_valid.shape}')

        np.save(f'train_data/{_dataset}_red/x_all', x_all, allow_pickle = False)
        np.save(f'train_data/{_dataset}_red/x_train', x_train, allow_pickle = False)
        np.save(f'train_data/{_dataset}_red/x_test', x_test, allow_pickle = False)
        np.save(f'train_data/{_dataset}_red/x_valid', x_valid, allow_pickle = False)

        np.save(f'train_data/{_dataset}_red/y_all', y_all, allow_pickle = False)
        np.save(f'train_data/{_dataset}_red/y_train', y_train, allow_pickle = False)
        np.save(f'train_data/{_dataset}_red/y_test', y_test, allow_pickle = False)
        np.save(f'train_data/{_dataset}_red/y_valid', y_valid, allow_pickle = False)
