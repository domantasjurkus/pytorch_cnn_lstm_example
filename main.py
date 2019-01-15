import torch

from dataset import SquaresDataset
from model import LSTMModel
import train_test_old
import train_test

FRAME_WIDTH = 10
LSTM_INPUT_SIZE = 50
LSTM_HIDDEN_SIZE = 32
BATCH_SIZE = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# generate sequence of frames
model = LSTMModel(LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE)
model = model.to(device)

# 
# Old way of generating datasetless examples
# 
# trainX, trainy = generate_examples(FRAME_WIDTH, 300)
# trainX, trainy = torch.Tensor(trainX), torch.Tensor(trainy)
# testX, testy = generate_examples(FRAME_WIDTH, 50)
# testX, testy = torch.Tensor(testX), torch.Tensor(testy)
# transpose from (samples, timestep, w, h, c) to (samples, timestep, c, h, w)
# trainX.transpose_(4, 2).transpose_(3, 4)
# testX.transpose_(4, 2).transpose_(3, 4)
# train_test_old.train(model, trainX, trainy, testX, testy, n_classes=2, epochs=1)

# 
# Data generation with dataset that will allow using batches
# 
squares_dataset_train = SquaresDataset(frame_width=FRAME_WIDTH, n=500)
squares_dataset_test = SquaresDataset(frame_width=FRAME_WIDTH, n=100)
xtion1_train_loader = torch.utils.data.DataLoader(squares_dataset_train, batch_size=BATCH_SIZE)
xtion1_test_loader = torch.utils.data.DataLoader(squares_dataset_test, batch_size=BATCH_SIZE)

train_test.train(model, xtion1_train_loader, xtion1_test_loader, n_classes=2, epochs=1)