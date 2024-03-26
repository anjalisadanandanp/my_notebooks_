import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt

def clones(module, N):
    """ Produce N identical layers """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    """ Mask out subsequent positions """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    mask = mask == 0
    return mask.bool()



class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % h == 0     # d_model = h * d_k

        #we assume d_v always equals d_k
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h
        linear_layer = nn.Linear(d_model, d_model, bias = True)
        self.linear_layers = clones(linear_layer, 4)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):

        """ Compute 'Scaled Dot Product Attention' """
        # query, key, value : [batch_size, head_num, seq_len, d_k]

        batch_size, head_num, seq_len, d_k = query.shape

        matmul = torch.matmul(query, key.transpose(-2, -1)) 
        scores = matmul / torch.sqrt(torch.tensor(d_k).float())

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_vals = nn.Softmax(dim=-1)(scores)

        # if dropout is not None:
        #     attention_vals = dropout(attention_vals)

        # print("Head 1: attention")
        # print(attention_vals[:, 0, :, :])
        # print("Head 2")
        # print(attention_vals[:, 1, :, :])
        # print("Head 3")
        # print(attention_vals[:, 2, :, :])
        # print("Head 4")
        # print(attention_vals[:, 3, :, :])

        output = torch.matmul(attention_vals, value)

        return output, attention_vals

    def forward(self, query, key, value, mask=None):

        batch_size, seq_len_q, d_model = query.shape
        query = query.view(batch_size, seq_len_q, self.h, self.d_k).transpose(1, 2)

        batch_size, seq_len_k_v, d_model = key.shape
        key = key.view(batch_size, seq_len_k_v, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len_k_v, self.h, self.d_k).transpose(1, 2)


        query = query.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_k*self.h)
        key = key.transpose(1, 2).contiguous().view(batch_size, seq_len_k_v, self.d_k*self.h)
        value = value.transpose(1, 2).contiguous().view(batch_size, seq_len_k_v, self.d_k*self.h)

        query = self.linear_layers[0](query)
        key = self.linear_layers[1](key)
        value = self.linear_layers[2](value)

        query = query.view(batch_size, seq_len_q, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len_k_v, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len_k_v, self.h, self.d_k).transpose(1, 2)

        x, attention_vals = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_k*self.h)
        x = self.linear_layers[3](x)

        del query, key, value

        return x



class PositionWiseFeedforward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedforward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_seq_len = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        PE = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)        # [max_seq_len, 1]
        i = torch.arange(0, d_model, 2)   
        div_term = torch.exp(i * (-torch.log(torch.tensor(10000.0)) / d_model)).view(1, -1)     # [1, d_model/2]
        angles = position*div_term
        PE[:, 0::2] = torch.sin(angles)
        PE[:, 1::2] = torch.cos(angles)
        PE = PE.unsqueeze(0)        # [1, max_seq_len, d_model]

        self.register_buffer('PE', PE)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x + self.PE[:, :seq_len, :].requires_grad_(False)
        x = self.dropout(x)
        return x



class Embeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.embedding(x) 
        x = x * torch.sqrt(torch.tensor(self.d_model).float())
        return x



class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(d_model))        
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        # a_2, b_2 are learnable parameters
        # a_2, b_2 : [GAIN, ADAPTIVE BIAS]
        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    


class SublayerConnection(nn.Module):
    
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        #sublayer : MultiHeadedAttention or PositionWiseFeedforward 

        return self.norm(x + self.dropout(sublayer(x)))



class EncoderLayer(nn.Module):

    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionWiseFeedforward(d_model, d_ff)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):

        #print("encoder self attention")
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask=mask))
        
        x = self.sublayer[1](x, self.feed_forward)

        return x



class Encoder(nn.Module):

    def __init__(self, d_model, h, d_ff, N, dropout=0.1):
        super(Encoder, self).__init__()

        self.layers = clones(EncoderLayer(d_model, h, d_ff, dropout), N)

    def forward(self, x, mask):
            
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x



class DecoderLayer(nn.Module):

    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(h, d_model)
        self.cross_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionWiseFeedforward(d_model, d_ff)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
        self.d_model = d_model

    def forward(self, x, memory, src_mask=None, tgt_mask=None):

        #print("decoder self attention")
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask=tgt_mask))

        #print("decoder cross attention")
        x = self.sublayer[1](x, lambda y: self.cross_attn(y, memory, memory, mask=src_mask))
        
        x = self.sublayer[2](x, self.feed_forward)

        return x



class Decoder(nn.Module):
    
    def __init__(self, d_model, h, d_ff, N, dropout=0.1):
        super(Decoder, self).__init__()

        self.layers = clones(DecoderLayer(d_model, h, d_ff, dropout), N)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):

        for layer in self.layers:
            x = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return x



class Generator(nn.Module):

    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()

        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
    


class EncoderDecoder(nn.Module):
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, position_encoding, generator, d_model):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.position_encoding = position_encoding
        self.generator = generator
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):

        src = self.src_embed(src)
        tgt = self.tgt_embed(tgt)

        src = self.position_encoding(src)
        tgt = self.position_encoding(tgt)

        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.generator(output)

        return output
    
    def encode(self, src, src_mask=None):
        src = self.src_embed(src)
        src = self.position_encoding(src)
        return self.encoder(src, mask=src_mask)
    
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        tgt = self.tgt_embed(tgt)
        tgt = self.position_encoding(tgt)
        return self.decoder(tgt, memory, src_mask=src_mask, tgt_mask=tgt_mask)
    



