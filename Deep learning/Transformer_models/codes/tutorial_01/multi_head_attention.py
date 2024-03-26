# MultiHead Attention

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()

        self.d_model = d_model      #input dimension of Q, K, V
        self.n_head = n_head        #number of heads
        self.d_k = d_model // n_head        # dimension of Q, K, V after linear projection

        # Note: d_model must be divisible by num_heads

        # The W_Q, W_K, W_V are the weight matrices for all the attention heads but represented as a single matrix (to ease the computation)

        # d_model = n_head * d_k
        self.W_Q = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        self.W_K = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        self.W_V = nn.Linear(in_features=d_model, out_features=d_model, bias=True)

        self.W_O = nn.Linear(in_features=d_model, out_features=d_model, bias=True)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        # Lets say we have query, key and value vectors 
        # This function computes the scaled dot product attention of these vectors
            
        # Q: [batch_size, n_head, sequence_length, d_k]
        # K: [batch_size, n_head, sequence_length, d_k]
        # V: [batch_size, n_head, sequence_length, d_k]

        # mask: [batch_size, n_head, sequence_length, sequence_length]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # mask scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)

        # output: [batch_size, n_head, sequence_length, d_k] 
        output = torch.matmul(attention, V)

        return output

    def split_heads(self, x):

        """ Returns x after splitting the last dimension (d_model) into (n_head, d_k) so that it can be processed by each heads"""

        # print("Before splitting heads: \n")
        #print("Input tensor x before splitting :\n", x)
        # x: [batch_size, sequence_length, d_model]
        batch_size, sequence_length, d_model = x.size()
        # print("Batch size:", batch_size, "Sequence length:", sequence_length, "d_model:", d_model)

        # print("After splitting heads: \n")
        # print("Batch size:", batch_size, "Sequence length:", sequence_length, "n_heads:", self.n_head, "d_k:", self.d_k)
        # print("The last dimension of the input is divided into n_heads and d_k")

        # splitting the last dimension into n_heads and d_k
        x = x.view(batch_size, sequence_length, self.n_head, self.d_k)

        x = x.transpose(1, 2)
        # print("shape after transposing", x.shape)
        return x
    
    def combine_heads(self, x):
        batch_size, n_heads, seq_length, d_k = x.size()

        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Q: [batch_size, sequence_length, d_model]
        # K: [batch_size, sequence_length, d_model]
        # V: [batch_size, sequence_length, d_model]

        # Linear projections of Q, K, V
        # Get the query, key and value vectors for all the heads by applying linear projection
        Q = self.W_Q(Q)         # Q: [batch_size, sequence_length, d_model]
        K = self.W_K(K)         # K: [batch_size, sequence_length, d_model]
        V = self.W_V(V)         # V: [batch_size, sequence_length, d_model]

        # Split the query, key and value vectors for all the heads for attention calculation
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot product attention calculation
        # Q, K, V: [batch_size, n_head, sequence_length, d_k]
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine the attention output for all the heads
        attention_output = self.combine_heads(attention_output)

        # Final linear projection
        output = self.W_O(attention_output)

        return output

attention_heads = MultiHeadAttention(d_model = 512, n_head = 8)

# random input for testing the model 
Q = torch.rand(64, 20, 512) # (batch_size, sequence_length, d_model)
K = torch.rand(64, 20, 512) # (batch_size, sequence_length, d_model)
V = torch.rand(64, 20, 512) # (batch_size, sequence_length, d_model)

# we pass the current values of Q, K, V to the forward function of the MultiHeadAttention class
output = attention_heads.forward(Q, K, V)

print("Shape of output:", output.shape)
# print("Output: \n", output)
