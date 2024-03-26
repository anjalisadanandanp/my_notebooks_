import torch
import torch.nn as nn
import numpy as np





class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()

        self.d_model = d_model      #input dimension of Q, K, V
        self.n_head = n_head        #number of heads
        self.d_k = d_model // n_head        # dimension of Q, K, V after linear projection

        # Note: d_model must be divisible by num_heads

        # The W_Q, W_K, W_V are the weight matrices for all the attention heads but represented as a single matrix (to ease the computation)

        # d_model = n_head * d_k
        self.W_Q = nn.Linear(in_features=d_model, out_features=d_model, bias=True).double()
        self.W_K = nn.Linear(in_features=d_model, out_features=d_model, bias=True).double()
        self.W_V = nn.Linear(in_features=d_model, out_features=d_model, bias=True).double()

        self.W_O = nn.Linear(in_features=d_model, out_features=d_model, bias=True).double()

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





class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff, bias=True).double()
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model, bias=True).double()

        self.relu = nn.ReLU()

    def forward(self, x):

        # x: [batch_size, sequence_length, d_model]
        # output: [batch_size, sequence_length, d_model]
        
        output = self.relu(self.linear_1(x))
        output = self.linear_2(output)

        return output
    




class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__() 
        self.multi_head_attention = MultiHeadAttention(d_model, n_head)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=d_model).double()
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=d_model).double()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        attn_out = self.multi_head_attention(x, x, x, mask)
        x = self.layer_norm_1(x + self.dropout(attn_out))
        ff_out = self.position_wise_feed_forward(x)
        x = self.layer_norm_2(x + self.dropout(ff_out))
        return x
    




class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__() 
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.cross_attn = MultiHeadAttention(d_model, n_head)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=d_model).double()
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=d_model).double()
        self.layer_norm_3 = nn.LayerNorm(normalized_shape=d_model).double()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.layer_norm_1(x + self.dropout(self_attn_out))
        cross_attn_out = self.cross_attn(x, encoder_out, encoder_out, src_mask)
        x = self.layer_norm_2(x + self.dropout(cross_attn_out))
        ff_out = self.position_wise_feed_forward(x)
        x = self.layer_norm_3(x + self.dropout(ff_out))
        return x





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

        encoding = torch.tensor(encoding)
        return encoding
    
    def forward(self, x):
        x = x + self.encoding
        return x
    



class Transformer(nn.Module):

    def __init__(self, d_model, n_head, d_ff, src_vocab_size, tgt_vocab_size, num_layers, max_seq_length, dropout):
        super().__init__()
        self.encoder_emdedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=d_model)
        self.decoder_embedding = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(max_seq_length, d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for layer in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout) for layer in range(num_layers)])

        self.linear = nn.Linear(in_features=d_model, out_features=tgt_vocab_size, bias=True).double()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)
        seq_length = tgt_mask.shape[-1]
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        input = self.encoder_emdedding(src)
        input = self.positional_encoding(input)
        input = self.dropout(input) 

        output = self.decoder_embedding(tgt)
        output = self.positional_encoding(output)
        output = self.dropout(output)

        for layer in self.encoder_layers:
            input = layer(input, src_mask)

        for layer in self.decoder_layers:
            output = layer(output, input, src_mask, tgt_mask)

        output = self.linear(output)
        output = self.softmax(output)
        return output
    



src_vocab_size = 5
tgt_vocab_size = 5
d_model = 4
n_heads = 2
num_layers = 6
d_ff = 20
max_seq_length = 4
dropout = 0.1

t_model = Transformer(d_model, n_heads, d_ff, src_vocab_size, tgt_vocab_size, num_layers, max_seq_length, dropout)

src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
tgt_data = torch.randint(1, src_vocab_size, (64, max_seq_length))


import torch.optim as optim
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(t_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

t_model.train()

for epoch in range(1000):
    optimizer.zero_grad()
    output = t_model(src_data, tgt_data)
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data.contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")