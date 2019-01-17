#!/usr/bin/env python

import sys
import torch
from torch.nn.functional import cosine_similarity as cos

def main(args):
    if len(args) < 1:
        sys.stderr.write("One required argument: <model file>\n")
        sys.exit(-1)

    model = torch.load(args[0])
    lookup = model.corpus.dictionary.word2idx
    emb = model.encoder

    pairs = [ ('king', 'queen'), 
              ('boston', 'philadelphia'), 
              ('king', 'automobile'), 
              ('car', 'automobile'),
              ('year', 'month'),
              ('year', 'country'),
              ('japan', 'country'),
              ('japan', 'china'),
              ('japan', 'germany') ]

    for pair in pairs:
        if pair[0] not in lookup:
            sys.stderr.write('%s not in this models dictionary\n' % (pair[0]))
            continue
        if pair[1] not in lookup:
            sys.stderr.write('%s not in this models dictionary\n' % (pair[1]))
            continue
        ind1 = torch.LongTensor([lookup[pair[0]]]).to('cuda')
        ind2 = torch.LongTensor([lookup[pair[1]]]).to('cuda')
        
        emb1 = emb(ind1)
        emb2 = emb(ind2)
        sim = cos(emb1, emb2)
        print("similarity between %s and %s is %f" % (pair[0], pair[1], sim.item()))

if __name__ == '__main__':
    main(sys.argv[1:])
