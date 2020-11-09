import os
import argparse

from _nn_architectures import LIST_OF_CLASSIFIERS
from _nn_architectures.utils import *

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose = False, validation = 'normal'):

    if classifier_name == 'fcn':
        from _nn_architectures import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes,
        verbose = verbose, validation = validation)
    if classifier_name == 'mlp':
        from _nn_architectures import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes,
        verbose = verbose, validation = validation)
    if classifier_name == 'tlenet':
        from _nn_architectures import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose = verbose, validation = validation)
    if classifier_name == 'mcdcnn':
        from _nn_architectures import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes,
        verbose = verbose, validation = validation)
    if classifier_name == 'cnn':
        from _nn_architectures import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes,
        verbose = verbose, validation = validation)
    if classifier_name == 'inception':
        from _nn_architectures import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes,
        verbose = verbose, validation = validation)

# Arguments to the function
parser = argparse.ArgumentParser()
parser.add_argument("classifier_name", help = "Name of the classifier that will be trained")
parser.add_argument("dataset", help = "Name of the training/testing dataset (system name)")
parser.add_argument("--validation", help = " - k-fold: performs k-fold cross-validation; - normal (default): uses training and testing datasets (a validation dataset is extracted by default)")
parser.add_argument("--folds", help = "number of folds (if performing k-fold cross-validation). It defaults to 5.", type = int)
parser.add_argument("--verbose", help = "verbosity (default 0 - not verbose; 1 - verbose)", type = bool)
args = parser.parse_args()

# Classifier name
_classifier = args.classifier_name
if _classifier not in LIST_OF_CLASSIFIERS:
    raise ValueError('Classifier not implemented')

# Dataset (system name)
_dataset = args.dataset

# Optional arguments
if args.validation:
    if args.validation not in ['k-fold', 'normal']:
        raise ValueError('Validation method not valid')
    else:
        _validation = args.validation
else:
    _validation = 'normal'

if args.folds:
    if args.folds <= 0:
        raise ValueError('Fold number not valid')
    else:
        _folds = args.folds
elif args.validation == 'k-fold':
    print('\nNo fold number specified. Set k = 5.\n')
    _folds = 5

if args.verbose:
    _verbose = args.verbose
else:
    _verbose = 0

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    # Checking output directory (creating it if it does not exist)
    _p_ds = os.path.join(os.getcwd(), "experiments", _dataset)
    if not os.path.exists(_p_ds): os.mkdir(_p_ds)

    output_directory = os.path.abspath(os.path.join(_p_ds, _classifier))
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    print(output_directory)

    print('{t:-^30}'.format(t = ''))
    print(f'Classifier: {_classifier}')
    print(f'Dataset: {_dataset}')
    print(f'Validation: {_validation}')
    if _validation == 'k-fold': print(f"Folds: {_folds}")
    print('{t:-^30}'.format(t = ''))

    # Loading Dataset
    data, input_shape, nb_classes = load_dataset(_dataset, _validation)

    # Extracting validation dataset
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    x_valid, y_valid = data['x_valid'], data['y_valid']
    x_all, y_all = data['x_valid'], data['y_valid']

    # Creating classifier
    classifier = create_classifier(_classifier, input_shape, nb_classes, output_directory,
    verbose = _verbose, validation = _validation)

    if _validation == 'normal':
        # 'Normal' training
        normal_training(classifier, x_train, y_train, x_test, y_test, x_valid, y_valid, nb_classes = nb_classes)
    else:
        # Running training with k-fold cross-validation
        k_fold_training(classifier, x_all, y_all, x_valid, y_valid, folds = _folds, nb_classes = nb_classes)
