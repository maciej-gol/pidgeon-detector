import cv2
import glob
import numpy as np
import random

from model import createModel
random.seed(128)


model1 = createModel()
model1.load_weights('weights.hdf5')

for fname in sorted(glob.glob('matches/non-pidgeon-32/*')):
    i = cv2.imread(fname)
    pred = model1.predict(np.array([i]))[0][0]
    if round(pred):
        print(fname, pred)
