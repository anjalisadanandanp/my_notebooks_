import torch
import torch.nn as nn
src_vocab_size = 100
tgt_vocab_size = 100
max_seq_length = 5

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(-tgt_vocab_size, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

print("(batch_size, seq_length):", src_data.shape)
print("(batch_size, seq_length):", tgt_data.shape)

print(tgt_data[0])


def generate_mask(src, tgt):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)
    tgt_mask = (tgt > 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, seq_length, 1)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    print(nopeak_mask.shape)
    print(tgt_mask.shape)
    print(tgt_mask[0])
    tgt_mask = tgt_mask & nopeak_mask
    print(tgt_mask.shape)
    print(tgt_mask[0])
    return src_mask, tgt_mask

src_mask, tgt_mask = generate_mask(src_data, tgt_data)
print("src_mask:", src_mask.shape)
print("tgt_mask:", tgt_mask.shape)
