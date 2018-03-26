import cv2
import glob
import numpy as np
import random

from model import createModel

random.seed(128)


def load_data():
    non_pidgeons = glob.glob('matches/no-pidgeon-32/*')
    pidgeons = glob.glob('matches/pidgeon-32/*')

    non_pidgeons = list(map(cv2.imread, non_pidgeons))
    pidgeons = list(map(cv2.imread, pidgeons))

    random.shuffle(non_pidgeons)
    random.shuffle(pidgeons)

    x_train = non_pidgeons[:2000] + pidgeons[:700]
    y_train = [0] * 2000 + [1] * 700

    x_test = non_pidgeons[2000:] + pidgeons[700:]
    y_test = [0] * (len(non_pidgeons) - 2000) + [1] * (len(pidgeons) - 700)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


train_data, train_labels_one_hot, test_data, test_labels_one_hot = load_data()

model1 = createModel()
batch_size = 128
epochs = 25
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

model1.evaluate(test_data, test_labels_one_hot)
model1.save_weights('weights.hdf5')
