import sys
sys.path.append("Deep learning/Transformer_models/codes/tutorial_02/model")

from transformer_modules import EncoderDecoder, Encoder, Decoder, Generator, MultiHeadedAttention, PositionWiseFeedforward, PositionalEncoding, Embeddings
from transformer_modules import subsequent_mask

import torch
import torch.nn as nn
import copy



def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
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


def inference_test():

    test_model = make_model(src_vocab = 100, tgt_vocab = 100)
    test_model.eval()

    src = torch.randint(100, (1, 20)).type(torch.LongTensor)
    print("Example Input:", src)
    print("Example Input Shape:", src.shape)
    src_mask = None

    target = torch.randint(100, (1, 20)).type(torch.LongTensor)
    print("Example Target:", target)
    print("Example Target Shape:", target.shape)

    tgt_mask = subsequent_mask(target.size(1)).type(torch.LongTensor)

    ys = test_model(src, target, src_mask=src_mask, tgt_mask=tgt_mask)

    print("Example Untrained Model Prediction:", ys)
    print("Example Untrained Model Prediction:", ys.shape)


def run_tests():

    for _ in range(10):
        inference_test()

run_tests()
