#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import pad_sequences

# XOR Truth Table
# x | y | Output
# 0 | 0 | 0
# 0 | 1 | 1
# 1 | 0 | 1
# 1 | 1 | 0
# Even parity bit: The number of 1's are counted. If odd, parity is 1. Even = 0


def generate_samples(length=50):
    ''' Generate random binary strings of variable length

    Args:
        length: Length of single binary string
    Returns:
        Numpy array of binary strings and array of parity bit labels
    '''
    # Generate random strings of length 50
    if length == 50:
        data = np.random.randint(2, size=(100000, length)).astype('float32')
        labels = [0 if sum(i) % 2 == 0 else 1 for i in data]
    # Generate random strings of variable length
    else:
        data = []
        labels = []
        for i in range(100000):
            # Choose random length
            length = np.random.randint(1, 51)
            data.append(np.random.randint(2, size=(length)).astype('float32'))
            labels.append(0 if sum(data[i]) % 2 == 0 else 1)
        data = np.asarray(data)
        # Pad binary strings with 0's to make sequence length same for all
        data = pad_sequences(data, maxlen=50, dtype='float32', padding='pre')

    labels = np.asarray(labels, dtype='float32')
    train_size = data.shape[0]
    size = int(train_size * 0.20)
    # Split data into train/test sets
    X_test = data[:size]
    X_train = data[size:]
    y_test = labels[:size]
    y_train = labels[size:]
    # Expand dimension to feed into LSTM layer
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    return X_train, y_train, X_test, y_test


def build_model():
    ''' Build LSTM model using Keras

    Args:
        None
    Returns:
        Compiled LSTM model
    '''
    model = Sequential()
    model.add(LSTM(32, input_shape=(50, 1)))
    model.add(Dense(1, activation='sigmoid'))
    # Display model summary
    model.summary()
    model.compile('adam', loss='binary_crossentropy', metrics=['acc'])

    return model


def plot_model(history):
    ''' Plot model accuracy and loss

    Args:
        history: Keras dictionary contatining training/validation loss/acc
    Returns:
        Plots model's training/validation loss and accuracy history
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    return


def main(length=50):
    ''' Train LSTM neural network to solve XOR problem '''
    X_train, y_train, X_test, y_test = generate_samples(length=length)
    model = build_model()
    history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                        validation_split=0.2, shuffle=False)

    # Evaluate model on test set
    preds = model.predict(X_test)
    preds = np.round(preds[:, 0]).astype('float32')
    acc = (np.sum(preds == y_test) / len(y_test)) * 100.
    print('Accuracy: {:.2f}%'.format(acc))

    # Plot model acc and loss
    plot_model(history)
    return


if __name__ == '__main__':
    ''' Execute main program '''
    # Set seed
    np.random.seed(21)
    # Grab user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', help='define binary string length (50 or -1)')
    args = parser.parse_args()
    if args.length == '50':
        print('Using randomly generated binary strings of constant length 50...')
        main(length=50)
    elif args.length == '-1':
        print('Using randomly generated binary strings of variable length between 1 and 50...')
        main(length=-1)
    else:
        print('Invalid length entry.')
