import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cnn_output_height = 1
        self.lstm_hidden_size  = 96
        self.lstm_num_layers  = 1
        self.num_classes       = 4

        self.conv1 = nn.Conv2d(3, 96, kernel_size=(5, 1), stride=1, padding='same')
        self.conv2 = nn.Conv2d(96, 96, kernel_size=(3, 1), stride=1, padding='same')
        self.freq_pooling1 = nn.AvgPool2d(kernel_size=(5, 1), stride=(4, 1))
        self.freq_pooling2 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.global_average_pooling = nn.AdaptiveAvgPool1d(2 * self.lstm_hidden_size)
        self.batch_norm  = nn.BatchNorm2d(96)

        self.lstm_input_feats_size = self.cnn_output_height * 96
        self.lstm = nn.LSTM(self.lstm_input_feats_size, self.lstm_hidden_size, self.lstm_num_layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.lstm_hidden_size * 2, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # (BatchSize, ch=3, 40, 40)
        batch_size = x.shape[0]
        #x = x.unsqueeze(1) # ch軸追加
        # conv_block1 = self.batch_norm(self.freq_pooling1(F.relu(self.conv1(x))))           # (BatchSize, ch=96, 8, 40)
        # conv_block2 = self.batch_norm(self.freq_pooling2(F.relu(self.conv2(conv_block1)))) # (BatchSize, ch=96, 4, 40)
        # conv_block3 = self.batch_norm(self.freq_pooling2(F.relu(self.conv2(conv_block2)))) # (BatchSize, ch=96, 2, 40)
        # conv_block4 = self.batch_norm(self.freq_pooling2(F.relu(self.conv2(conv_block3)))) # (BatchSize, ch=96, 1, 40)

        conv_block1 = self.freq_pooling1(torch.relu(self.conv1(x)))           # (BatchSize, ch=96, 8, 40)
        conv_block2 = self.freq_pooling2(torch.relu(self.conv2(conv_block1))) # (BatchSize, ch=96, 4, 40)
        conv_block3 = self.freq_pooling2(torch.relu(self.conv2(conv_block2))) # (BatchSize, ch=96, 2, 40)
        conv_block4 = self.freq_pooling2(torch.relu(self.conv2(conv_block3))) # (BatchSize, ch=96, 1, 40)

        reshape_layer = conv_block4.reshape(batch_size, conv_block4.shape[3], self.lstm_input_feats_size) # (BatchSize, length=40, num_feats=96)
        lstm_out, (hidden, cell) = self.lstm(reshape_layer)
        lstm_out = self.global_average_pooling(lstm_out)
        fc_out = self.fc(lstm_out[:,-1,:])
        out = self.softmax(fc_out)
   
        return out