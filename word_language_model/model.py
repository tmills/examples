import torch.nn as nn
from torch.autograd import Variable
from lcrnn import LcRnn
from srnn import StochasticRnn
import numpy as np

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        elif rnn_type=='NLC':
            self.rnn = LcRnn(ninp, num_layers=nlayers, hidden_size=nhid)
        elif rnn_type=='SRNN':
            self.rnn = StochasticRnn(ninp, num_layers=nlayers, hidden_size=nhid)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'NLC', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_params = sum([np.prod(p.size()) for p in trainable_parameters])
        print("Model initialized with %d trainable parameters" % (num_params))

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.parsing = False

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        if not self.parsing:
            output = self.decoder(output[:,:,:self.nhid])
            # output = self.decoder(output.permute(1,0,2)[:,:,:self.nhid])
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        elif self.rnn_type == 'NLC':
            return self.rnn.init_hidden(bsz)
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def update_callback(self, epochs, batch):
        try:
            self.rnn.update_callback(epochs, batch)
        except:
            # Do nothing
            pass
