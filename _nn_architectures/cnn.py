# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc

from .utils.utils import save_logs
from .utils.utils import calculate_metrics

class Classifier_CNN:

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
            self.verbose = verbose

            self.model.save_weights(os.path.join(self.output_directory, 'model_init.hdf5'))

        return

    def build_model(self, input_shape, nb_classes):
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60: # for italypowerondemand dataset
            padding = 'same'

        conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        output_layer = keras.layers.Dense(units=nb_classes,activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        return model

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

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 10,
            min_lr = 0.0001)

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath = file_path_best, monitor = 'val_loss',
            save_best_only = True, verbose = self.verbose)

        early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min',
            verbose = self.verbose)

        self.callbacks = [reduce_lr, model_checkpoint, early_stopping]

        # -------------------------------
        # Training model
        # -------------------------------

        # x_val and y_val are only used to monitor the test loss and NOT for training
        mini_batch_size = 16
        nb_epochs = 1000

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

        save_logs(self.output_directory, hist, y_pred, y_true, duration, lr = False, fold = fold)

        keras.backend.clear_session()

    def predict(self, x_test, y_test, x_train, y_train, nb_classes, return_df_metrics = True):

        t_0_pred = time.time()

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
