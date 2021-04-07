from torch import log_softmax
import torch.nn as nn

class CNNModel(nn.Module):

    def __init__(self, num_classes=5):
        super(CNNModel, self).__init__()

        self.convolution1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout2d(0.2)

        self.convolution2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.linear1 = nn.Linear(in_features=21632, out_features=2048)
        self.linear2 = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, input):
        conv1_output = self.convolution1(input)
        relu1_output = self.relu1(conv1_output)
        max_output = self.max_pool(relu1_output)

        dropout_output = self.dropout(max_output)

        conv2_output = self.convolution2(dropout_output)
        relu2_output = self.relu2(conv2_output)

        relu2_output = relu2_output.view(-1, 21632)
        linear1_output = self.linear1(relu2_output)
        linear2_output = self.linear2(linear1_output)
        
	final_output = log_softmax(linear2_output, dim=1)

        return final_output
