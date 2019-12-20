import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical  # convert to one-hot-encoding
from sklearn.metrics import confusion_matrix

from src.classifier.classification_preparation import read_file_classes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.ERROR)

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap("Blues")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_history(evolv_history):
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(evolv_history.history['loss'], color='b', label="Training loss")
    ax[0].plot(evolv_history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(evolv_history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(evolv_history.history['val_accuracy'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)


def CNN_model(X_train, l_classes):
    the_model = Sequential()

    the_model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu', input_shape=X_train.shape[1:]))
    the_model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu'))
    the_model.add(MaxPool2D(pool_size=(2, 2)))
    the_model.add(Dropout(0.25))

    the_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
    the_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
    the_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    the_model.add(Dropout(0.25))

    the_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
    the_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
    the_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    the_model.add(Dropout(0.25))

    the_model.add(Flatten())
    the_model.add(Dense(256, activation="relu"))
    the_model.add(Dropout(0.5))
    the_model.add(Dense(len(l_classes), activation="softmax"))

    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    the_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return the_model


def load_data(pickle_path, l_classes):
    pickle_in_file = open(pickle_path, "rb")
    pickle_in = pickle.load(pickle_in_file)
    x_data = pickle_in["X"]
    y_data = pickle_in["Y"]
    y_data = to_categorical(y_data, num_classes=len(l_classes))
    return x_data / 255.0, y_data


if __name__ == '__main__':
    # --- CONST for training --- #
    epochs = 8
    # epochs = 20
    batch_size = 16
    # batch_size = 32
    validation_split = 0.1

    # --- PATH --- #
    # dataset_path = "/media/remi/hdd_linux/DataSet/mnist_numeric/"
    dataset_path = "/media/hdd_linux/DataSet/Mine/"
    dataset_file = os.path.join(dataset_path, "mnist_numeric.pickle")
    names_file = os.path.join(dataset_path, "data.names")
    models_path = 'src/classifier/models/'

    # --- LOADING --- #
    list_classes = read_file_classes(names_file)
    X, Y = load_data(dataset_file, list_classes)

    # --- Training --- #
    model = CNN_model(X_train=X, l_classes=list_classes)  # Model Generation

    history = model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_split=validation_split)

    new_model_name = "{}model_{}.h5".format(models_path, len(os.listdir(models_path)))
    model.save(new_model_name)

    # Results Analysis
    nbr_of_test_sample = round(validation_split * len(Y))

    # Predict the values from the validation classifier
    Y_pred = model.predict(X[:-nbr_of_test_sample])
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y[:-nbr_of_test_sample], axis=1)
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes=list_classes)
    plot_history(history)
    plt.show()
    # CF https://github.com/jessestauffer/MNIST-CNN-Keras/blob/master/mnist.py