#!/usr/bin/env python
# coding: utf-8
import argparse
import time
import math
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import data
import model
from data import batchify, get_batch, add_unk, read_glove

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--prefix-len', type=int, default=3,
                    help='prefix length of sentences to train on')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size')
# parser.add_argument('--bptt', type=int, default=35,
#                     help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--load', type=str,  default=None,
                    help='path to load pre-trained model')
parser.add_argument('--unk', action="store_true",
                    help='Replace rare words with \'unk\' randomly to train an unknown word embedding')
parser.add_argument('--glove', type=str, default=None, required=False,
                    help='Path to Glove embeddings file if pre-trained embeddings are used.')
                    
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data, max_seq_len=args.prefix_len)
embeddings = None
if not args.glove is None:
    embeddings = read_glove(args.glove, corpus.dictionary)

if not embeddings is None:
    first_tok = next(iter(embeddings))
    if len(embeddings[first_tok]) != args.emsize:
        print("ERROR: Embedding size (--emsize) %d is not the same as pre-trained embedding size %d" % (len(embeddings[first_tok]), args.emsize))
        sys.exit(-1)

eval_batch_size = 100
device = torch.device("cuda" if args.cuda else "cpu")
train_data = batchify(corpus.train, args.batch_size, device)
val_data = batchify(corpus.valid, eval_batch_size, device)
test_data = batchify(corpus.test, eval_batch_size, device)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.load is None:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, corpus=corpus, embeddings=embeddings)
else:
    with open(args.load, 'rb') as f:
        model = torch.load(f)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def report_time(start_time, name):
    length = time.time() - start_time
    print('Running %s took %d')

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    for sent_len in data_source.keys():
        for batch, i in enumerate(range(0, data_source[sent_len].size(1) - 1, eval_batch_size)):
            data, targets = get_batch(data_source[sent_len], i, eval_batch_size, prefix_len=sent_len-1, evaluation=True)
            actual_batch_size = data.shape[1]
            hidden = model.init_hidden(actual_batch_size)
            output, hidden = model(data, hidden)

            flat_dim = (sent_len-1) * actual_batch_size
            total_loss += ( criterion(output.view(flat_dim,-1), targets.contiguous().view(flat_dim)).data / sent_len)

            # hidden = repackage_hidden(hidden)
    return total_loss.item()

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    
    sent_lens = list(train_data)
    random.shuffle(sent_lens)
    num_seqs = 0
    for sent_len in sent_lens:
        for batch, i in enumerate(range(0, train_data[sent_len].size(1) - 1, args.batch_size)):
            # print(model.rnn.cell.w_f.weight)
            data, targets = get_batch(train_data[sent_len], i, args.batch_size, prefix_len=sent_len-1)
            actual_batch_size = data.shape[1]
            if args.unk:
                data = add_unk(data, corpus)

            # For the last batch the batch size may be smaller:
            hidden = model.init_hidden(actual_batch_size)
            model.zero_grad()
            output, hidden = model(data, hidden)
            flat_dim = actual_batch_size*(sent_len-1)
            loss = criterion(output.view(flat_dim, -1), targets.contiguous().view(flat_dim))
            loss.backward()

            # Haven't seen any benefit but this would go here:
            # torch.nn.utils.clip_grad_norm_(model.parameters(),0.1)
            optimizer.step()

            total_loss += loss.data
            num_seqs += data.shape[1]

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss.item() / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr (ADAM) | ms/batch {:5.2f} | '
                        'loss {:5.2f} | {:5d} sequences | ppl NA'.format(
                    epoch, batch, len(train_data) // args.prefix_len,
                    elapsed * 1000 / args.log_interval, cur_loss, num_seqs))# , math.exp(cur_loss)))
                model.update_callback(epoch, batch)
                total_loss = 0
                start_time = time.time()

    model.epoch_callback(epoch, args.epochs)
    return num_seqs

# Loop over epochs.
# lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        # print(model.rnn.cell.w_f.weight)
        epoch_start_time = time.time()
        seqs_processed = train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | num seqs {:5d} '
                'valid ppl NA'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, seqs_processed)) #, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl NA'.format(
    test_loss))
print('=' * 89)
