import torch

from dataset import SquaresDataset
from model import LSTMModel
import train_test_old
import train_test

FRAME_WIDTH = 10
LSTM_INPUT_SIZE = 50
LSTM_HIDDEN_SIZE = 32
BATCH_SIZE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LSTMModel(LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE)
model = model.to(device)

squares_dataset_train = SquaresDataset(frame_width=FRAME_WIDTH, n=1000)
squares_dataset_test = SquaresDataset(frame_width=FRAME_WIDTH, n=100)
xtion1_train_loader = torch.utils.data.DataLoader(squares_dataset_train, batch_size=BATCH_SIZE)
xtion1_test_loader = torch.utils.data.DataLoader(squares_dataset_test, batch_size=BATCH_SIZE)

train_test.train(model, xtion1_train_loader, xtion1_test_loader, n_classes=2, epochs=2)