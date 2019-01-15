import numpy as np
from numpy import zeros
from random import randint
from random import random
from matplotlib import pyplot

from model import LSTMModel
from util import *
from train_test import *

HIDDEN_SIZE = 32

# generate sequence of frames
size = 10
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