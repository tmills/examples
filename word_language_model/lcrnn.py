#!./env/bin/python

import torch
from torch import nn, FloatTensor, ByteTensor, LongTensor
from torch.nn.functional import sigmoid
from torch.autograd import Variable, Function
import numpy as np

class LcRnnCell(nn.Module):
    def __init__(self, input_size, depth=1, hidden_size=100):
        super(LcRnnCell, self).__init__()

        if depth != 1:
            raise NotImplementedError("Left-corner RNN only works with one layer right now.")

        self.depth = depth
        self.embed_size = input_size
        self.hidden_size = hidden_size
        # 2 hidden variables, 4 possibile outcomes
        self.depth_size = 2 * 4 * self.hidden_size
        self.stride = 2*self.hidden_size*(self.depth+1)

        ## A given 00 (Shain et al.)
        ## P θ A (a_t | b^{d-1}_t−1 a^d_t−1 )
        ## Ignore b^{d-1} for now
        ## Plus the word (new)
        self.w_a00 = nn.Linear(self.embed_size + self.hidden_size * 2, self.hidden_size, bias=False)

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

        # B given 01 (ibid)
        ## P θ B (b^{d−1}_t | b^{d-1}_{t−1} a^d_{t−1})
        self.w_b01 = nn.Linear(self.embed_size + self.hidden_size * 2, self.hidden_size, bias=False)

        # Attention model: Given all a/b parameters: a/b (*2), each depth (+1 for zero depth), for 
        # all possible f/j configurations at each time step (*4)
        # And predict a 4-way probability distribution over f/j configs:
        self.attention = nn.Linear(self.depth_size * self.depth, 4, bias=False)

        # A few static matrices that we use to map from our attention states to
        # a mask that will select out the hidden state for the highest probability
        # state.
        self.selection_striper = nn.Parameter(torch.zeros(4, 4*self.stride), requires_grad=False)
        self.selection_striper[0, :self.stride] = 1
        self.selection_striper[1, self.stride:2*self.stride] = 1
        self.selection_striper[2, 2*self.stride:3*self.stride] = 1
        self.selection_striper[3, 3*self.stride:] = 1

        self.stripe_expander = nn.Parameter(torch.cat( (torch.eye(self.stride), torch.eye(self.stride), torch.eye(self.stride), torch.eye(self.stride)), 0), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters
        """
        self.w_a00.reset_parameters()
        self.w_a10.reset_parameters()
        self.w_b00.reset_parameters()
        self.w_b11.reset_parameters()
        self.w_b10.reset_parameters()
        self.w_b01.reset_parameters()


    def forward(self, X, hidden=None):
        
        prev_a, prev_b, prev_depth, _ = hidden
        batch_size = X.shape[0]
        hidden_size = prev_b.shape[-1]

        batch_range = range(batch_size)

        # Depth "0" is initialized to 0 (needed for conditioning of depth 1)
        ab_00 = [ Variable(torch.zeros(batch_size, 2*hidden_size).cuda()) ]
        ab_01 = [ Variable(torch.zeros(batch_size, 2*hidden_size).cuda()) ]
        ab_10 = [ Variable(torch.zeros(batch_size, 2*hidden_size).cuda()) ]
        ab_11 = [ Variable(torch.zeros(batch_size, 2*hidden_size).cuda()) ]

        for d in range(1, self.depth+1):
            fork_join_a = prev_a[:,d,:]
            nofork_nojoin_a = self.w_a00(torch.cat( (X, prev_b[:,d-1,:], prev_a[:,d,:]), 1))
            fork_nojoin_a = self.w_a10(torch.cat( (X, prev_b[:,d-1,:]), 1))
            nofork_join_a = prev_a[:,d-1,:]

            ## At next depth, need to update a and/or b
            next_a_d_00 = (torch.eq(prev_depth, float(d)).float() * nofork_nojoin_a
                        +    # at shallower depth, copy over
                        torch.gt(prev_depth, float(d)).float() * prev_a[:,d,:]
                        +    # at deeper depth, zero out
                        torch.lt(prev_depth, float(d)).float() * torch.zeros_like(prev_a[:,d,:]))
            
            next_a_d_11 = (torch.eq(prev_depth, float(d)).float() * fork_join_a
                        +    # at shallower depth, copy over
                        torch.gt(prev_depth, float(d)).float() * prev_a[:,d,:]
                        +    # at deeper depth, zero out
                        torch.lt(prev_depth, float(d)).float() * torch.zeros_like(prev_a[:,d,:]))

            next_a_d_10 = (torch.eq(prev_depth+1, float(d)).float() * fork_nojoin_a
                        +    # at shallower depth, copy over
                        torch.gt(prev_depth+1, float(d)).float() * prev_a[:,d,:]
                        +    # at deeper depth, zero out
                        torch.lt(prev_depth+1, float(d)).float() * torch.zeros_like(prev_a[:,d,:]))

            next_a_d_01 = (torch.eq(prev_depth-1, float(d)).float() * nofork_join_a
                        +    # at shallower depth, copy over
                        torch.gt(prev_depth-1, float(d)).float() * prev_a[:,d,:]
                        +    # at deeper depth, zero out
                        torch.lt(prev_depth-1, float(d)).float() * torch.zeros_like(prev_a[:,d,:]))
            

            fork_join_b = self.w_b11( torch.cat( ( X, prev_b[:,d,:]), 1) )
            nofork_nojoin_b = self.w_b00( torch.cat( (X, prev_a[:,d,:], next_a_d_00), 1))
            fork_nojoin_b = self.w_b10( torch.cat( (X, next_a_d_10), 1))
            nofork_join_b = self.w_b01( torch.cat( (X, prev_b[:,d-1,:], prev_a[:,d,:]), 1))

            next_b_d_00 = (torch.eq(prev_depth, float(d)).float() * nofork_nojoin_b
                        +    # at shallower depth, copy over
                        torch.gt(prev_depth, float(d)).float() * prev_b[:,d,:]
                        +    # at deeper depth, zero out
                        torch.lt(prev_depth, float(d)).float() * torch.zeros_like(prev_b[:,d,:]))
            next_b_d_11 = (torch.eq(prev_depth, float(d)).float() * fork_join_b
                        +    # at shallower depth, copy over
                        torch.gt(prev_depth, float(d)).float() * prev_b[:,d,:]
                        +    # at deeper depth, zero out
                        torch.lt(prev_depth, float(d)).float() * torch.zeros_like(prev_b[:,d,:]))
            next_b_d_10 = (torch.eq(prev_depth+1, float(d)).float() * fork_nojoin_b
                        +    # at shallower depth, copy over
                        torch.gt(prev_depth+1, float(d)).float() * prev_b[:,d,:]
                        +    # at deeper depth, zero out
                        torch.lt(prev_depth+1, float(d)).float() * torch.zeros_like(prev_b[:,d,:]))
            next_b_d_01 = (torch.eq(prev_depth-1, float(d)).float() * nofork_join_b
                        +    # at shallower depth, copy over
                        torch.gt(prev_depth-1, float(d)).float() * prev_b[:,d,:]
                        +    # at deeper depth, zero out
                        torch.lt(prev_depth-1, float(d)).float() * torch.zeros_like(prev_b[:,d,:]))

            next_ab_00 = torch.cat( (next_a_d_00, next_b_d_00), 1)
            next_ab_01 = torch.cat( (next_a_d_01, next_b_d_01), 1)
            next_ab_10 = torch.cat( (next_a_d_10, next_b_d_10), 1)
            next_ab_11 = torch.cat( (next_a_d_11, next_b_d_11), 1)

            ab_00.append(next_ab_00)
            ab_01.append(next_ab_01)
            ab_10.append(next_ab_10)
            ab_11.append(next_ab_11)
            
        
        # Now flatten the depth and predict the attention variables:
        # next_state_flat = torch.squeeze( next_state.view(batch_size, 1, -1) )
        ab_00_flat = torch.stack(ab_00, 1).view(batch_size, 1, -1)
        ab_01_flat = torch.stack(ab_01, 1).view(batch_size, 1, -1)
        ab_10_flat = torch.stack(ab_10, 1).view(batch_size, 1, -1)
        ab_11_flat = torch.stack(ab_11, 1).view(batch_size, 1, -1)

        next_state_flat = torch.squeeze( torch.cat( (ab_00_flat, ab_01_flat, ab_10_flat, ab_11_flat),  2) )

        # At time 0, and only at time 0, prev_Depth is 0, so we must choose 1/0
        mask = Variable(torch.ones(batch_size, 4).cuda())
        # if we're at depth 0, we can only allow 1/0
        mask[:, (0,1,3)] *= (1-torch.eq(prev_depth,0).float())
        # if we're at depth d, we cannot allow 1/0
        mask[:, (2,)] *= (1 - torch.eq(prev_depth, self.depth).float())

        att_vars = (mask * 
                    torch.nn.functional.softmax( self.attention( next_state_flat[:, self.depth_size:] ) ) )

        selection = torch.sign(att_vars - torch.unsqueeze(att_vars.max(1)[0],1) * Variable(torch.ones(1,4).cuda()) + np.finfo(np.double).tiny) + 1
        # Make sure this computation works without fail:
        assert selection.sum().data.cpu().numpy()[0] == selection.shape[0]
        # selection = Variable(torch.zeros( batch_size, 4, 1).cuda())
        # selection[ range(batch_size), MaxST(att_vars).data, : ] = 1
        # selection[ range(batch_size), torch.max(att_vars), : ] = 1

        striped = torch.mm(selection, self.selection_striper)
        ## It's ok up to here. Now we need to broadcast a dot product for each
        ## stripe across the stacked identity matrix to mask out the unwanted
        ## parts of the state space

        mask_list = []
        for b in range(batch_size):
            batch_mask = []
            for ind in range(self.stride):
                batch_mask.append(striped[b] * self.stripe_expander[:,ind])
            mask_list.append( torch.stack(batch_mask, 1) )
        striped_identity = torch.stack(mask_list, 0)

        # It's ok below here. 
        hidden = torch.squeeze(torch.bmm(torch.unsqueeze(next_state_flat,1), striped_identity))
        # hidden = torch.mm(striped, masked_)

        # stride = (self.depth+1) * self.hidden_size * 2
        # hidden = (selection[:,0] * next_state_flat[:, :stride] 
        #          + selection[:,1] * next_state_flat[:, stride:stride*2]
        #          + selection[:,2] * next_state_flat[:, stride*2:stride*3]
        #          + selection[:,3] * next_state_flat[:, stride*3:stride*4])
        
        # now hidden needs to be re-viewed as batch x depth x hidden
        hidden = hidden.view(batch_size, self.depth+1, self.hidden_size * 2)
        next_a = hidden[:,:, :self.hidden_size]
        next_b = hidden[:,:, self.hidden_size:]                 

        # compute the selected next depth and equivalent f/j variables from the selection:
        next_depth = torch.unsqueeze( (selection[:,0].long() * prev_depth[:,0]
                     + selection[:,1].long() * (prev_depth[:,0]-1)
                     + selection[:,2].long() * (prev_depth[:,0]+1)
                     + selection[:,3].long() * prev_depth[:,0]), 1)
        f = torch.unsqueeze( (selection[:,2] * 1 + selection[:,3] * 1), 1)
        j = torch.unsqueeze( (selection[:,1] * 1 + selection[:,3] * 1), 1)

        return next_a, next_b, next_depth, (f, j)
        

class LcRnn(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, cat_layers=False):
        super(LcRnn, self).__init__()

        self.embed_size = input_size
        # we're splitting our hidden state into 2 so make sure it's even:
        if not hidden_size % 2 == 0:
            hidden_size += 1
        self.hidden_size = (hidden_size-2) // 2
        self.depth = num_layers
        self.hidden_a_init = FloatTensor(1, self.depth+1, self.hidden_size).zero_().cuda()
        self.hidden_b_init = FloatTensor(1, self.depth+1, self.hidden_size).zero_().cuda()
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
                    Variable(LongTensor(batch_size,1).zero_().cuda(), requires_grad=False),
                    None)

    @staticmethod
    def _forward_rnn(cell, X, hx):
        seq_len = X.size(0)
        output = []
        for step in range(seq_len):
            a_next, b_next, depth_next, (f,j) = cell.forward(X[step], hx)
            hx_next = (a_next, b_next, depth_next, (f,j))
            # Right now we take the highest depth as the output -- may eventually want to 
            # cat together hidden variables at all depth levels (see cta_layers option in contsructor)
            output.append(torch.cat((a_next[:,-1,:], b_next[:,-1,:], f.float(), j.float()), 1))
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
            hx = (hx, hx, None)
        else:
            hx = hidden

        layer_output = None
        layer_output, (last_a, last_b, last_depth, fj) = LcRnn._forward_rnn(
                cell=self.cell, X=X, hx=hx)

        return layer_output, (last_a, last_b, fj)

def equals(variable, val):
    if not len(variable.shape) == 1 or not variable.shape[0] == 1:
        raise Exception("The equals() method can only be called on ByteTensors with one value.")
    
    return variable.cpu()[0].data[0] == val