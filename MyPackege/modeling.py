#####################################
    # Modeling
#####################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
import matplotlib.pyplot as plt
import os


def build_model(hidden_layer_units: list, dropouts: list) -> keras.models.Sequential():
    if len(hidden_layer_units) != len(dropouts):
        raise ValueError("Length of 'units' and 'dropouts' should be same!")
    else:
        model = keras.models.Sequential()
        for ii in range(len(hidden_layer_units)):
            model.add(keras.layers.Dense(hidden_layer_units[ii], activation='relu', kernel_initializer=keras.initializers.he_normal(seed=ii)))   # He Weight Initialization
            if dropouts[ii] > 0:
                model.add(keras.layers.Dropout(dropouts[ii]))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
        return model


class Model_builder:

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, path_for_result: str):
        # fix random seed for reproducibility
        np.random.seed(10)
        # Set dataset
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.result_path = path_for_result
        print('Finished setting datasets!')

    def standardize(self, except_col: list):
        # Normalization (Mean=0, Standard division=1)
        standard = StandardScaler()
        if except_col == None:
            standard.fit(self.X_train)
            self.X_train = standard.transform(self.X_train)
            self.X_test = standard.transform(self.X_test)
        elif len(except_col) > 0:
            standard.fit(self.X_train[self.X_train.columns[self.X_train.columns != except_col]])
            self.self.X_train[self.X_train.columns[self.X_train.columns != except_col]] = \
                standard.transform(self.X_train[self.X_train.columns[self.X_train.columns != except_col]])
            self.X_test[self.X_test.columns[self.X_test.columns != except_col]] = \
                standard.transform(self.X_test[self.X_test.columns[self.X_test.columns != except_col]])
        else:
            raise ValueError("Select except_col in 'None' or list format")

    def holdout_modeling(self, units: list, dropout_ratios: list, patience: int, epochs: int, batch_size:int, split_ratio: float):
        model = build_model(hidden_layer_units=units, dropouts=dropout_ratios)
        # Avoid overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
        )
        # Fitting
        self.history = model.fit(
            self.X_train,
            self.y_train,
            validation_split=split_ratio,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=early_stopping
        )
        self.batch_size = batch_size
        self.model = model

    def save_history(self):
        # Plot training & validation accuracy values
        plt.figure()
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Hold-out method')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join(self.result_path, 'Holdout_acc.png'))
        plt.close()
        # Plot training & validation loss values
        plt.figure()
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join(self.result_path, 'Holdout_loss.png'))
        plt.close()

    def show_train_result(self):
        # Evaluate the model on the training data using `evaluate`
        results = self.model.evaluate(self.X_train, self.y_train, batch_size=self.batch_size, verbose=0)
        print("train loss, train acc:", results)

    def save_test_result(self):
        path = os.path.join(self.result_path, 'submission.csv')
        self.y_pred = self.model.predict(self.X_test)
        output = pd.DataFrame({'ID': self.X_test.PassengerId, '取引価格（総額）_log': self.y_pred})
        output.to_csv(path, index=False)
        print("Your submission was successfully saved!")