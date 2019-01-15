import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, lstm_input_size=50, lstm_hidden_size=32):
        super(LSTMModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, bias=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
            num_layers=1, batch_first=True)
        self.classifier = nn.Linear(32, 2)

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        # expect x.shape = (samples, timesteps, channels, height, width)
        samples, timesteps, c, h, w = x.size()
        c_in = x.view(samples*timesteps, c, h, w)
        c_out = self.features(c_in)

        r_in = c_out.view(samples, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)

        classes = self.classifier(r_out[:, -1, :])
        softmax = F.log_softmax(classes, dim=1)
        return softmax