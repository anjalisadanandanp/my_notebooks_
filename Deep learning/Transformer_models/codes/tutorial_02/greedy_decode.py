
import torch   
import copy
import torch.nn as nn

import sys
sys.path.append("Deep learning/Transformer_models/codes/tutorial_02/model")

from transformer_modules import subsequent_mask
from transformer_modules import EncoderDecoder, Encoder, Decoder, Generator, MultiHeadedAttention, PositionWiseFeedforward, PositionalEncoding, Embeddings
from torch.autograd import Variable


state_dict = torch.load("Deep learning/Transformer_models/codes/tutorial_02/trained_copy_model/model.pt")

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=64, d_ff=64*4, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionWiseFeedforward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(d_model, h, d_ff, N, dropout),
        Decoder(d_model, h, d_ff, N, dropout),
        Embeddings(d_model, src_vocab),
        Embeddings(d_model, tgt_vocab), 
        position,
        Generator(d_model, tgt_vocab),
        d_model
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

model = make_model(src_vocab = 12, tgt_vocab = 12)

model.load_state_dict(state_dict)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode( Variable(ys), memory, src_mask, 
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


model.eval()
# src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]))
src = Variable(torch.LongTensor([[1,5,7,4,5,3,7,8,1,2]]) )
src_mask = (src != 0).unsqueeze(-2).unsqueeze(-3)
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))





