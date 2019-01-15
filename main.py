import numpy as np
from numpy import zeros
from random import randint
from random import random
from matplotlib import pyplot
# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import TimeDistributed

from model import LSTMModel
from util import *
from train_test import *

HIDDEN_SIZE = 32

# generate sequence of frames
size = 10
# frames, right = build_frames(size)
model = LSTMModel(50, HIDDEN_SIZE)
model = model.to(device)

trainX, trainy = generate_examples(size, 300)
trainX, trainy = torch.Tensor(trainX), torch.Tensor(trainy)
testX, testy = generate_examples(size, 50)
testX, testy = torch.Tensor(testX), torch.Tensor(testy)

# transpose from (samples, timestep, w, h, c) to (samples, timestep, c, h, w)
trainX.transpose_(4, 2).transpose_(3, 4)
testX.transpose_(4, 2).transpose_(3, 4)

train(model, trainX, trainy, testX, testy, n_classes=2, epochs=1)

# hamsterX, hamstery = generate_examples(size, 3)
# hamsterX, hamstery = torch.Tensor(hamsterX), torch.Tensor(hamstery)
# hamsterX.transpose_(4, 2).transpose_(3, 4)
# out = model(hamsterX)
# print(hamsterX[:, -1])
# print(out)