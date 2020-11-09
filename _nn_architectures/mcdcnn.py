# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
from .utils.utils import save_logs
from .utils.utils import calculate_metrics
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc

class Classifier_MCDCNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True, validation = 'normal'):

        self.validation = validation
        self.verbose = verbose

        # Output directory depending on validation
        if self.validation == 'normal':
            self.output_directory = output_directory
        elif self.validation == 'k-fold':
            # Creating the output directory in case k-fold cross-validation is required
            self.output_directory = os.path.join(output_directory, 'k-fold')
            if not os.path.exists(self.output_directory):
                os.mkdir(self.output_directory)

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()

            self.model.save_weights(os.path.join(self.output_directory, 'model_init.hdf5'))
        return

    def build_model(self, input_shape, nb_classes):
        n_t = input_shape[0]
        n_vars = input_shape[1]

        padding = 'valid'

        if n_t < 60: # for ItalyPowerOndemand
            padding = 'same'

        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            input_layer = keras.layers.Input((n_t,1))
            input_layers.append(input_layer)

            conv1_layer = keras.layers.Conv1D(filters=8,kernel_size=5,activation='relu',padding=padding)(input_layer)
            conv1_layer = keras.layers.MaxPooling1D(pool_size=2)(conv1_layer)

            conv2_layer = keras.layers.Conv1D(filters=8,kernel_size=5,activation='relu',padding=padding)(conv1_layer)
            conv2_layer = keras.layers.MaxPooling1D(pool_size=2)(conv2_layer)
            conv2_layer = keras.layers.Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        if n_vars == 1:
            # to work with univariate time series
            concat_layer = conv2_layers[0]
        else:
            concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)

        fully_connected = keras.layers.Dense(units=732,activation='relu')(concat_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=0.0005),
                      metrics=['accuracy'])

        return model

    def prepare_input(self,x):
        new_x = []
        n_t = x.shape[1]
        n_vars = x.shape[2]

        for i in range(n_vars):
            new_x.append(x[:,:,i:i+1])

        return  new_x

    def fit(self, x_train, y_train, x_val, y_val, fold = ''):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # Loading initial model (for k-fold cross-validation: to avoid "cheating" the method)
        # Ref: https://stackoverflow.com/questions/42052576/k-fold-cross-validation-initialise-network-after-each-fold-or-not
        self.model.load_weights(os.path.join(self.output_directory, 'model_init.hdf5'))

        # Path to save the model
        if self.validation == 'normal':
            # Path to save model
            file_path_best = os.path.join(self.output_directory, 'best_model.hdf5')
            file_path_last = os.path.join(self.output_directory, 'last_model.hdf5')
        elif self.validation == 'k-fold':
            print('{t:-^30}'.format(t = ''))
            print(f'Fold: {fold}')
            print('{t:-^30}'.format(t = ''))

            file_path_best = os.path.join(self.output_directory, f'{fold}_best_model.hdf5')
            file_path_last = os.path.join(self.output_directory, f'{fold}_last_model.hdf5')

        self.path_best = file_path_best
        self.path_last = file_path_last

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.5, patience = 10,
            min_lr = 0.0001)

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath = file_path_best, monitor = 'val_loss',
            save_best_only=True, verbose = self.verbose)

        early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min',
            verbose = self.verbose)

        self.callbacks = [reduce_lr, model_checkpoint, early_stopping]

        # -------------------------------
        # Training model
        # -------------------------------

        mini_batch_size = 16
        nb_epochs = 60

        #x_train, x_val, y_train, y_val = \
        #    train_test_split(x, y, test_size=0.33)

        x_train = self.prepare_input(x_train)
        x_val = self.prepare_input(x_val)

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(file_path_last)

        model = keras.models.load_model(file_path_best)

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis = 1)
        y_true = np.argmax(y_val, axis = 1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration, fold = fold)

        keras.backend.clear_session()

    def predict(self, x_test, y_test, x_train, y_train, nb_classes, return_df_metrics = True):

        t_0_pred = time.time()

        x_test = self.prepare_input(x_test)

        model_path = self.path_best
        model = keras.models.load_model(model_path)
        y_logits = model.predict(x_test)

        t_f_pred = time.time() - t_0_pred

        if return_df_metrics:
            y_pred = np.argmax(y_logits, axis = 1)
            y_true = np.argmax(y_test, axis = 1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            df_metrics['prediction_time'] = t_f_pred
            return df_metrics
        else:
            return y_logits
