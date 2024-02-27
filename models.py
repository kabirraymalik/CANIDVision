import torch
import torch.nn as nn
import torchvision

class VideoClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VideoClassifier, self).__init__()
        self.cnn = torchvision.models.resnet50()
        self.cnn.fc = nn.Identity()  # Removing the last fully connected layer
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes) #only using the last time sequence frame to perform classification

    def forward(self, x):
        # x shape: (batch_size, sequence_length, C, H, W)
        batch_size, sequence_length, C, H, W = x.size()
        c_out = torch.empty(batch_size, sequence_length, 2048).to(x.device)  # Placeholder for CNN features, 2048 for resnet-50 input
        for t in range(sequence_length):
            c_out[:, t, :] = self.cnn(x[:, t, :, :, :])  # Perform feature extraction by passing each frame through the CNN
        lstm_out, (h_n, c_n) = self.lstm(c_out)  # passing final frame through LSTM
        out = self.fc(lstm_out[:, -1, :])  # Classify based on the last output of LSTM
        return out