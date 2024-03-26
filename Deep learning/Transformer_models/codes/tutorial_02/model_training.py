#   COPY TASK   
import wandb
from wandb import AlertLevel
from datetime import timedelta
import torch.optim.lr_scheduler as lr_scheduler

config = dict()
config["learning_rate"] = 0.5
config["num_batches"] = 2000
config["VOCAB_SIZE"] = 12
config["threshold"] = 1e-3
config["batch_size"] = 512
config["seq_length"] = 10
config["start_token_val"] = 0
config["end_token_val"] = 11
config["step_size"] = 250
config["gamma"] = 0.85
config["num_encoder_decoder_layers"] = 6
config["num_heads"] = 8
config["d_model"] = 64
config["d_ff"] = 64*4
config["dropout"] = 0.1
config["print_result_at"] = 10


# start a new wandb run to track this script
run = wandb.init(

    # set the wandb project where this run will be logged
    project="transformer_sequence_copy_model",
    
    # track hyperparameters and run metadata
    config=config,

    id = "model_training_v1.1"

)

import torch   
import copy
import torch.nn as nn

import sys
sys.path.append("Deep learning/Transformer_models/codes/tutorial_02/model")

from transformer_modules import subsequent_mask
from transformer_modules import EncoderDecoder, Encoder, Decoder, Generator, MultiHeadedAttention, PositionWiseFeedforward, PositionalEncoding, Embeddings


def make_model(
    src_vocab, tgt_vocab, 
    N=config["num_encoder_decoder_layers"], 
    d_model=config["d_model"], d_ff=config["d_ff"], 
    h=config["num_heads"],
    dropout=config["dropout"]):

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

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model




def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )




class Batch:

    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt, mask_no_val = 0):
        self.src = src
        self.src_mask = (src != mask_no_val).unsqueeze(-2).unsqueeze(-3)

        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, mask_no_val)

    @staticmethod
    def make_std_mask(tgt, mask_no_val=0):
        """Create a mask to hide mask_no_val and future words."""
        tgt_mask = (tgt != mask_no_val).unsqueeze(-2).unsqueeze(-3)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask




def data_gen(start_token_val, max_token_val, batch_size, sequence_length, num_batches):
    """Generate random data for a src-tgt copy task."""
    for i in range(num_batches):    
        data = torch.randint(start_token_val+1, max_token_val, size=(batch_size, sequence_length))
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src=src, tgt=tgt, mask_no_val=0)





from torch.autograd import Variable
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode( Variable(ys), memory, src_mask, 
                           Variable(subsequent_mask(ys.size(1)).type_as(src_mask.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys






data = data_gen(start_token_val=config["start_token_val"], max_token_val=config["end_token_val"], batch_size=config["batch_size"], sequence_length=config["seq_length"], num_batches=config["num_batches"])




def run_epoch(data_iter):

    for id, batch in enumerate(data_iter):

        print("Batch ID:", id)

        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

        #convert the target tokens to probabilities using one-hot encoding
        tgt_probs = torch.zeros(batch.tgt_y.shape[0], batch.tgt_y.shape[1], VOCAB_SIZE)
        for i in range(batch.tgt_y.shape[0]):
            for j in range(batch.tgt_y.shape[1]):
                tgt_probs[i][j][batch.tgt_y[i][j]] = 1
                tgt_probs = tgt_probs.requires_grad_(False).clone().detach()

        #print("Target probs:", tgt_probs)

        #zero the gradients
        optimizer.zero_grad()

        loss = criterion(out, tgt_probs)

        wandb.log({"loss": loss})
        wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})

        loss.backward()
        optimizer.step()
        scheduler.step()

        if id % config["print_result_at"] == 0:

            print("\n")
            print("Batch ID:", id, "Loss:", loss.item(), "Learning rate:", optimizer.param_groups[0]['lr'])

            # set the model to eval mode
            model.eval()

            # Decoding and printing the output of a sequence
            src = Variable(torch.LongTensor([batch.src[-1].numpy()]))      # one example from the training set
            src_mask = (src != 0).unsqueeze(-2).unsqueeze(-3)
            with torch.no_grad():   
                print("Source(UNknown) :", src)     
                print("Decoded(UNknown):", greedy_decode(model, src, src_mask, max_len=config["seq_length"], start_symbol=src[0,0].item()))

            src = Variable(torch.LongTensor([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]))
            src_mask = (src != 0).unsqueeze(-2).unsqueeze(-3)
            with torch.no_grad():   
                print("Source(known) :", src)     
                print("Decoded(known):", greedy_decode(model, src, src_mask, max_len=config["seq_length"], start_symbol=src[0,0].item()))
       
            # set the model back to train mode
            model.train()

        if loss < threshold:
            wandb.alert(
                title='Loss decreased below threshold',
                text=f'Loss {loss} is below the acceptable threshold {threshold}',
                level=AlertLevel.WARN,
                wait_duration=timedelta(minutes=5)
            )

    return




VOCAB_SIZE = config["VOCAB_SIZE"]
threshold = config["threshold"]
model = make_model(src_vocab = VOCAB_SIZE, tgt_vocab = VOCAB_SIZE)

resume = False

if resume == True:
    model.load_state_dict(torch.load("Deep learning/Transformer_models/codes/tutorial_02/trained_copy_model/model.pt"))


criterion = nn.KLDivLoss(reduction="sum", log_target=False)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9)

# scheduler = lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

scheduler = lr_scheduler.LambdaLR(
            optimizer=optimizer, 
            lr_lambda=lambda step: rate(step, model_size=config["d_model"], factor=1.0, warmup=200))

model.train()
run_epoch(data)

#save the trained model
torch.save(model.state_dict(), "Deep learning/Transformer_models/codes/tutorial_02/trained_copy_model/model.pt")
