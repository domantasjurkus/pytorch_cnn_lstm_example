import torch
from torch.utils.data import Dataset

import util

class SquaresDataset(Dataset):
    def __init__(self, frame_width=10, n=300):
        self.trainX, self.trainy = util.generate_examples(frame_width, n)
        self.trainX, self.trainy = torch.Tensor(self.trainX), torch.tensor(self.trainy)

    def __len__(self):
        return len(self.trainy)

    def __getitem__(self, index):
        return (self.trainX[index], self.trainy[index])