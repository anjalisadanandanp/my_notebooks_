import torch
import torch.nn as nn

# Compared to a standard feedforward network, which applies the same linear transformation to all input features, 
# the Position-wise Feedforward Network  (PFFN) applies a unique set of linear transformations to each position in the sequence. 

class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff, bias=True)
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):

        # x: [batch_size, sequence_length, d_model]
        # output: [batch_size, sequence_length, d_model]
        
        output = self.relu(self.linear_1(x))
        output = self.linear_2(output)

        return output
    

# run the forward pass of the PositionWiseFeedForward module
batch_size = 64   # batch_size is the number of sequences in the batch
sequence_length = 20    # sequence_length is the number of tokens in the sequence
d_model = 512   # d_model is the dimensionality of the input and output of the PFFN
d_ff = 2048   # d_ff is the number of neurons in the hidden layer of the PFFN

x = torch.rand(batch_size, sequence_length, d_model)

pffn = PositionWiseFeedForward(d_model, d_ff)
print(pffn.forward(x))