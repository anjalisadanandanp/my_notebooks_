import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, seq_length, d_model):
        super().__init__()
        self.encoding = self.positional_encoding(seq_length, d_model)

    def positional_encoding(self, seq_length, d_model):
        encoding = np.zeros((seq_length, d_model))
        for pos in range(seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    encoding[pos, i] = np.sin(pos / 10000**(2*i/d_model))
                else:
                    encoding[pos, i] = np.cos(pos / 10000**((2*i-1)/d_model))
        return encoding
    
    def forward(self, x):
        x = x + self.encoding
        return x

# Example usage
seq_length = 20
d_model = 512
pos_encoding = PositionalEncoding(seq_length=seq_length, d_model=d_model)
print(pos_encoding.encoding.shape)


plt.figure(figsize=(12,8))
plt.pcolormesh(pos_encoding.encoding, cmap='viridis')
plt.xlabel('Embedding Dimensions')
plt.ylabel('Token Position')
plt.colorbar()
plt.show()

#test forward
x = torch.zeros((5, seq_length, d_model))
x = pos_encoding(x)
# print(x)