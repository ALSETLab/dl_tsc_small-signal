import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc

from .utils.utils import save_logs
from .utils.utils import calculate_metrics
from .utils.utils import save_test_duration

class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64, lr=0.001,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs= 750,
                 validation = 'normal'):

        self.validation = validation

        # Output directory depending on validation
        if self.validation == 'normal':
            self.output_directory = output_directory
        elif self.validation == 'k-fold':
            # Creating the output directory in case k-fold cross-validation is required
            self.output_directory = os.path.join(output_directory, 'k-fold')
            if not os.path.exists(self.output_directory):
                os.mkdir(self.output_directory)

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(os.path.join(self.output_directory, 'model_init.hdf5'))

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, x_val, y_val, fold = ''):
        if not tf.test.is_gpu_available:
            print('error no gpu')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

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

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
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

        model_path = self.path_best
        model = keras.models.load_model(model_path)
        y_logits = model.predict(x_test, batch_size = self.batch_size)

        t_f_pred = time.time() - t_0_pred

        if return_df_metrics:
            y_pred = np.argmax(y_logits, axis = 1)
            y_true = np.argmax(y_test, axis = 1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            df_metrics['prediction_time'] = t_f_pred
            return df_metrics
        else:
            return y_logits
