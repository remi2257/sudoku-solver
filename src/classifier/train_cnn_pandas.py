import itertools
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical  # convert to one-hot-encoding
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.classifier.classification_preparation import read_file_classes

# import matplotlib.image as mpimg
# import seaborn as sns

# np.random.seed(2)


handwritten = False
# include_zero = handwritten
include_zero = True
nbr_classes = 9 + int(include_zero)
if handwritten:
    epochs = 4
    batch_size = 86
else:
    epochs = 10  # 20
    batch_size = 32  # 86

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the classifier
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the classifier
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=20,  # randomly rotate images_test in the range (degrees, 0 to 180)
    zoom_range=[0.9, 1.2],  # Randomly zoom image
    width_shift_range=0.2,  # randomly shift images_test horizontally (fraction of total width)
    height_shift_range=0.07,  # randomly shift images_test vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images_test
    vertical_flip=False)  # randomly flip images_test

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
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


def plot_history(history):
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)


def CNN_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(nbr_classes, activation="softmax"))

    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def load_data_split(data_path, handwritten=False):
    train_path = os.path.join(data_path, "mnist_train.csv")
    test_path = os.path.join(data_path, "mnist_test.csv")
    if handwritten:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    else:
        train = pd.read_csv(train_path, index_col=0)
        test = pd.read_csv(test_path, index_col=0)

    train = train.sample(frac=1).reset_index(drop=True)

    xtrain = train.drop(labels=["label"], axis=1)
    ytrain = train["label"] - min(train["label"].values)

    xtrain = xtrain.values.reshape(-1, 28, 28, 1)
    ytrain = to_categorical(ytrain, num_classes=nbr_classes)

    test = test.sample(frac=1).reset_index(drop=True)

    xtest = test.drop(labels=["label"], axis=1)
    ytest = test["label"] - min(test["label"].values)

    xtest = xtest.values.reshape(-1, 28, 28, 1)

    ytest = to_categorical(ytest, num_classes=nbr_classes)

    return xtrain / 255.0, ytrain, xtest / 255.0, ytest


def load_data(data_path, handwritten=False):
    dataset_path = os.path.join(data_path, "minst_train_test.csv")
    if handwritten:
        train = pd.read_csv(dataset_path)
    else:
        train = pd.read_csv(dataset_path, index_col=0)

    train = train.sample(frac=1).reset_index(drop=True)

    xtrain = train.drop(labels=["label"], axis=1)
    ytrain = train["label"] - min(train["label"].values)

    xtrain = xtrain.values.reshape(-1, 28, 28, 1)
    ytrain = to_categorical(ytrain, num_classes=nbr_classes)

    return xtrain / 255.0, ytrain


def main():
    train = True
    already_trained = True
    data_augmentation = True
    history = None
    if handwritten:
        dataset_path = "/media/hdd_linux/DataSet/mnist_handwritten/"

    else:
        # dataset_path = "/media/hdd_linux/DataSet/mnist_numeric/"
        dataset_path = "/media/hdd_linux/DataSet/Mine/"

    X, Y = load_data(dataset_path, handwritten)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)  # , random_state=2

    if not already_trained:
        models_path = 'model/'
        model = CNN_model()

    else:
        from tensorflow.keras.models import load_model
        models_path = 'model/'
        model_name = "{}my_model.h5".format(models_path, len(os.listdir(models_path)))
        model = load_model(model_name)
        print("Model Loaded")

    if train:
        if not data_augmentation:
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                validation_data=(x_test, y_test))  # validation_data=(X_val, Y_val)

        else:
            datagen.fit(x_train)
            # Fit the model
            history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                          epochs=epochs, validation_data=(x_test, y_test),
                                          verbose=1, steps_per_epoch=x_train.shape[0] // batch_size
                                          , callbacks=[learning_rate_reduction])

        new_model_name = "{}model_{}.h5".format(models_path, len(os.listdir(models_path)))
        model.save(new_model_name)
    classes = read_file_classes(dataset_path + "data.names")

    if train:
        # Predict the values from the validation classifier
        Y_pred = model.predict(x_test)
        # Convert predictions classes to one hot vectors
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        # Convert validation observations to one hot vectors
        Y_true = np.argmax(y_test, axis=1)
        # compute the confusion matrix
        confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    else:
        # Predict the values from the validation classifier
        Y_pred = model.predict(X)
        # Convert predictions classes to one hot vectors
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        # Convert validation observations to one hot vectors
        Y_true = np.argmax(Y, axis=1)
        # compute the confusion matrix
        confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
        count_wrong = 0
        for img, y_true, y_pred in zip(X, Y_true, Y_pred_classes):
            if y_true == y_pred:
                continue
            count_wrong += 1
            cv2.imshow("{} - Pred {} - True {}".format(count_wrong, classes[y_pred], classes[y_true]), img)
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes=classes)
    if train:
        plot_history(history)
    plt.show()

    # else:
    #     from keras.models import load_model
    #
    #     models_path = 'model/'
    #     new_model_name = "{}model_{}.h5".format(models_path, len(os.listdir(models_path)) - 1)
    #
    #     print(y_train[7000])
    #     plt.imshow(x_train[7000].reshape(28, 28))
    #     print(x_train[0][:,:,0])
    #     plt.show()


if __name__ == '__main__':
    main()
