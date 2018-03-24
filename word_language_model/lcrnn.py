#!./env/bin/python

import torch
from torch import nn, FloatTensor, ByteTensor, LongTensor
from torch.nn.functional import sigmoid
from torch.autograd import Variable

class BinaryStochasticNeuron(nn.Linear):
    def __init__(self, input_size):
        super(BinaryStochasticNeuron,self).__init__(input_size, 1, bias=False)
    
    def forward(self,X):
        return super(BinaryStochasticNeuron,self).forward(X)
    
    ## TOD: Implement backward and see if it's called
    def backward(self):
        return super(BinaryStochasticNeuron,self).backward()

class LcRnnCell(nn.Module):
    def __init__(self, input_size, depth=1, hidden_size=100):
        super(LcRnnCell, self).__init__()

        if depth != 1:
            raise NotImplementedError("Left-corner RNN only works with one layer right now.")

        self.depth = depth
        self.embed_size = input_size
        self.hidden_size = hidden_size

        ## Internal state has a value for f, j, vectors for a and b
        # self.internal_state_size = 2 + 2 * self.hidden_size

        ## f is dependent on b^d_t-1
        ## in 1 layer model, this is the last hidden_size values in the hidden state
        self.w_f = nn.Linear(self.hidden_size, 1, bias=False)
        ## if f is 0, j is dependent on a^d_t-1 and b^{d-1}_{t-1}, a from prev
        ## time step at this depth, and b at prev time step at higher depth
        # self.w_j0 = nn.Linear(self.hidden_size*2, 1, bias=False)
        ## if f is 1, j is dependent on b^d_{t-1}, b from prev time 
        ## step and same depth
        # self.w_j1 = nn.Linear(self.hidden_size, 1, bias=False)

        ## A given 00 (Shain et al.)
        ## P θ A (a_t | b^{d-1}_t−1 a^d_t−1 )
        ## Ignore b^{d-1} for now
        ## Plus the word (new)
        self.w_a00 = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size, bias=False)

        ## A given 10:
        ## P θ (a^d+1 | b^d_{t-1} p_t )
        ## Use word instead of pos and b from previous time step and above (the thing that's expanding)
        self.w_a10 = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size, bias=False)
        
        ## if fj is 11, a is the same at this depth. 
        ## if fj is 10, a is the same but there would be a new a at a lower depth
        ## if fj is 01, this depth wouldn't matter anymore
        
        ## B given 00 (ibid)
        ## P θ B (b_t | a^d_t, a^d_t−1 )
        ## Plus the word (new)
        self.w_b00 = nn.Linear(self.embed_size + self.hidden_size * 2, self.hidden_size, bias=False)

        ## B given 11 (ibid)
        ## P θ B (b t | b^d_{t−1} )
        ## Plus the word (new)
        self.w_b11 = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size, bias=False)

        ## B given 10 (ibid)
        ## P θ (b^{d+1} | a^{d+1}_t p_t )
        ## Use word instead of POS and a that was just gneerated
        self.w_b10 = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters
        """
        self.w_f.reset_parameters()
        # self.w_j0.reset_parameters()
        # self.w_j1.reset_parameters()
        self.w_a00.reset_parameters()
        self.w_a10.reset_parameters()
        self.w_b00.reset_parameters()
        self.w_b11.reset_parameters()
        self.w_b10.reset_parameters()


    def forward(self, X, hidden=None):
        
        prev_a, prev_b, prev_depth = hidden
        batch_size = X.shape[0]
        hidden_size = prev_b.shape[-1]

        batch_range = range(batch_size)

        if (prev_depth==-1).all():
            f = Variable(torch.ones(batch_size,1).cuda().long())
            j = Variable(torch.zeros(batch_size,1).cuda().long())
        else:
            f = torch.round(sigmoid(self.w_f(prev_b[batch_range,prev_depth[:,0],:]))).long()
            ## For now ignore j weights, always do f=j
            j = torch.add(torch.zeros_like(f), f)

        ## Inputs only depend on previous time step's depth:
        next_depth = torch.add(prev_depth, f.sub(j))

        next_a = []
        next_b = []
        for d in range(self.depth):
            fork_join_a = prev_a[:,d,:]
            nofork_nojoin_a = self.w_a00(torch.cat( (X, prev_a[:,d,:]), 1))
            ## At next depth, need to update a and/or b
            next_a_d = (torch.eq(next_depth, float(d)).float() *
                            ( ( (f * j).float() * fork_join_a ) +       # f=j=1
                              ( (1-f) * (1-j) ).float() * nofork_nojoin_a) #f=j=0
                        +    # at shallower depth, copy over
                        torch.lt(next_depth, float(d)).float() * prev_a[:,d,:]
                        +    # at deeper depth, zero out
                        torch.gt(next_depth, float(d)).float() * torch.zeros_like(prev_a[:,d,:]))
            
            fork_join_b = self.w_b11( torch.cat( ( X, prev_b[:,d,:]), 1) )
            nofork_nojoin_b = self.w_b00( torch.cat( (X, prev_a[:,d,:], next_a_d), 1))
            next_b_d = (torch.eq(next_depth, float(d)).float() *
                            ( ( (f * j).float() * fork_join_b) +
                              ( (1-f) * (1-j) ).float() * nofork_nojoin_b)
                        + # At shallower depth, copy over
                        torch.lt(next_depth, float(d)).float() * prev_b[d]
                        +   # at deeper depth, zero out
                        torch.gt(next_depth, float(d)).float() * torch.zeros_like(prev_a[:,d,:]))
            next_a.append(next_a_d)
            next_b.append(next_b_d)
        
        
        # ## Shape of a and b is (batches, depth, hidden_size)
        # next_a = torch.add(torch.zeros_like(prev_a), prev_a)
        # next_b = torch.add(torch.zeros_like(prev_b), prev_b)
        # for batch_ind in batch_range:
        #     X_word = torch.unsqueeze(X[batch_ind], 0)
        #     prev_word_depth = prev_depth[batch_ind].detach()
        #     next_word_depth = next_depth[batch_ind].detach()
        #     # if equals(f[batch_ind,0], 0) and equals(j[batch_ind,0], 0):
        #     if (f[batch_ind,0].data==0).all() and (j[batch_ind,0].data==0).all():
        #         ## a and b at current depth change:
        #         a00_input_vec = torch.cat((X_word, prev_a[batch_ind][prev_word_depth]), 1)
        #         next_a[batch_ind,next_word_depth.cpu().data[0]] = self.w_a00(a00_input_vec)
        #         b00_input_vec = torch.cat((X_word, prev_a[batch_ind][prev_word_depth], next_a[batch_ind][prev_word_depth]), 1)                
        #         next_b[batch_ind,next_word_depth.cpu().data[0]] = self.w_b00(b00_input_vec)
        #     elif (f[batch_ind,0].data==1).all() and (j[batch_ind,0].data==1).all():
        #         ## everything stays the same except b[d]
        #         b11_input_vec = torch.cat((X_word, prev_b[batch_ind][prev_word_depth]), 1)
        #         next_b[batch_ind,next_word_depth.cpu().data[0]] = self.w_b11(b11_input_vec)
        #     elif (f[batch_ind,0].data==1).all() and (j[batch_ind,0].data==0).all():
        #         ## generate new A at lower level, which in turn generates a b
        #         ## This is also used for the first word of a sentence (even if depth=1)
        #         assert (prev_depth[batch_ind].data==-1).all()

        #         # if equals(prev_word_depth, -1):
        #         ## First word in the sentence:
        #         a10_input_vec = torch.cat((X_word, Variable(torch.zeros(1,hidden_size).cuda())), 1)
        #         # else:
        #         #     raise NotImplementedError("Not allowed to make depth transitions yet")

        #         next_a[batch_ind,next_word_depth.data[0]] = self.w_a10(a10_input_vec)
        #         b10_input_vec = torch.cat((X_word, torch.unsqueeze(next_a[batch_ind,next_word_depth.data[0]], 0)), 1)
        #         next_b[batch_ind,next_word_depth.data[0]] = self.w_b10(b10_input_vec)

        #     else:
        #         raise NotImplentedError("This version only works with depth 1, so f=j=0 or f=j=1")

        return torch.stack(next_a, 1), torch.stack(next_b, 1), next_depth
        

class LcRnn(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, cat_layers=False):
        super(LcRnn, self).__init__()

        self.embed_size = input_size
        # we're splitting our hidden state into 2 so make sure it's even:
        if not hidden_size % 2 == 0:
            hidden_size += 1
        self.hidden_size = hidden_size // 2
        self.depth = num_layers
        self.hidden_a_init = FloatTensor(1, self.depth, self.hidden_size).zero_().cuda()
        self.hidden_b_init = FloatTensor(1, self.depth, self.hidden_size).zero_().cuda()
        self.cell = LcRnnCell(self.embed_size, hidden_size=self.hidden_size, depth=self.depth)
        if cat_layers:
            raise NotImplementedError("Cat'ing layers together is not yet supported. The representation will be the hidden states at the highest level.")
        self.cat_layers = cat_layers

        self.reset_parameters()


    def reset_parameters(self):
        self.cell.reset_parameters()
    
    def init_hidden(self, batch_size):
        # if batch_size != 1:
        #     raise NotImplementedError("This model only works with batch size 1 currently.")
        return (Variable(self.hidden_a_init.expand(batch_size, -1, -1)),
                    Variable(self.hidden_b_init.expand(batch_size, -1, -1)),
                    Variable(LongTensor(batch_size,1).zero_().cuda()-1, requires_grad=False))

    @staticmethod
    def _forward_rnn(cell, X, hx):
        seq_len = X.size(0)
        output = []
        for step in range(seq_len):
            a_next, b_next, depth_next = cell.forward(X[step], hx)
            hx_next = (a_next, b_next, depth_next)
            # Right now we take the highest depth as the output -- may eventually want to 
            # cat together hidden variables at all depth levels (see cta_layers option in contsructor)
            output.append(torch.cat((a_next[:,-1,:], b_next[:,-1,:]), 1))
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
            hx = Variable(X.data.new(batch_size, self.hidden_size).zero_())
            hx = (hx, hx)
        else:
            hx = hidden

        layer_output = None
        layer_output, (last_a, last_b, last_depth) = LcRnn._forward_rnn(
                cell=self.cell, X=X, hx=hx)

        return layer_output, (last_a, last_b)

def equals(variable, val):
    if not len(variable.shape) == 1 or not variable.shape[0] == 1:
        raise Exception("The equals() method can only be called on ByteTensors with one value.")
    
    return variable.cpu()[0].data[0] == val