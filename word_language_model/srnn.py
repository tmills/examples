#!./env/bin/python

import torch
from torch import nn, FloatTensor, ByteTensor, LongTensor
from torch.nn.functional import sigmoid
from torch.autograd import Variable

class StochasticCell(nn.Module):
    def __init__(self, input_size, layers=1, hidden_size=100):
        super(StochasticCell, self).__init__()

        if layers != 1:
            raise NotImplementedError("Left-corner RNN only works with one layer right now.")

        self.embed_size = input_size
        self.hidden_size = hidden_size

        self.w_f = nn.Linear(self.embed_size, 1, bias=False)
        self.w_a0 = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size, bias=False)
        self.w_a1 = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters
        """
        self.w_f.reset_parameters()
        self.w_a0.reset_parameters()
        self.w_a1.reset_parameters()

    def forward(self, X, hidden=None):
        
        batch_size = X.shape[0]

        f = torch.round(sigmoid(self.w_f(X)))

        input_vec = torch.cat((X, hidden[0]), 1)
        next_hidden = torch.unsqueeze(
                        f * self.w_a1(input_vec) + 
                        (1-f) * self.w_a0(input_vec),
                        0)
        # next_hidden = torch.zeros_like(hidden)

        # for batch_ind in range(batch_size):
        #     X_word = torch.unsqueeze(X[batch_ind], 0)
        #     input_vec = torch.cat((X_word, hidden[:,batch_ind]), 1)
        #     if (f[batch_ind] == 0).all():
        #         next_hidden[:,batch_ind,:] = self.w_a0(input_vec)
        #     else:
        #         next_hidden[:,batch_ind,:] = self.w_a1(input_vec)
            
        return next_hidden

class StochasticRnn(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(StochasticRnn, self).__init__()

        self.embed_size = input_size
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.cell = StochasticCell(self.embed_size, hidden_size=self.hidden_size, layers=self.layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.cell.reset_parameters()
    
    @staticmethod
    def _forward_rnn(cell, X, hx):
        seq_len = X.size(0)
        output = []
        for step in range(seq_len):
            hx_next = cell.forward(X[step], hx)
            output.append(hx_next[-1,:,:])
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx


    def forward(self, X, hidden, length=None):
        seq_len, batch_size, _ = X.size()

        if length is None:
            length = Variable(torch.LongTensor([seq_len] * batch_size))
            if X.is_cuda:
                device = X.get_device()
                # length = length.cuda(device)
        
        if hidden is None:
            hx = Variable(torch.zeros(batch_size, self.hidden_size))
        else:
            hx = hidden

        layer_output = None
        layer_output, hidden = StochasticRnn._forward_rnn(
                cell=self.cell, X=X, hx=hx)

        return layer_output, hidden

                