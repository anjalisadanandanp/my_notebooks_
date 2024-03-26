import torch
import torch.nn as nn

# Define the RNN model

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the input layer
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        output = self.i2o(combined)
        hidden = self.i2h(combined)
        output = self.log_softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    

#checking the RNN model
my_network = RNN(input_size=128, hidden_size=64, output_size=128)

#Do a forward pass
input = torch.randn(1, 128)
hidden = my_network.initHidden()
output, next_hidden = my_network(input, hidden)
print("Output:", output.shape)
print("Next hidden:", next_hidden.shape)
