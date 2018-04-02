
import sys
from  data import Corpus, batchify, get_batch
import torch
from torch.autograd import Variable
import argparse

def main(args):
    parser = argparse.ArgumentParser(description='Neural left corner parser')
    parser.add_argument('--prefix-len', type=int, default=3,
                        help='prefix length of sentences to train on')
    parser.add_argument('--data', type=str, default='./data/simple',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, required=True,
                        help='Saved model file to use for parsing')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')

    args = parser.parse_args(args)

    prefix_len = args.prefix_len + 1

    corpus = Corpus(args.data, seq_len=prefix_len)
    d = corpus.dictionary.idx2word
    val_data = batchify(corpus.valid, 1, prefix_len=prefix_len, cuda=args.cuda)
    eval_batch_size = 1

    with open(args.model, 'rb') as f:
        model = torch.load(f)
        model.parsing = True
        if args.cuda:
            model.cuda()
   
    model.eval()
    ntypes = len(corpus.dictionary)

    for i in range(0, val_data.size(1) - 1, eval_batch_size):
        hidden = model.init_hidden(eval_batch_size)
        data, targets = get_batch(val_data, i, eval_batch_size, prefix_len=prefix_len, evaluation=True)
        output, hidden = model(data, hidden)
        for batch_ind in range(eval_batch_size):
            for ind in range(prefix_len-1):
                if ind+1 < prefix_len:
                    fj_str = "%d/%d" % (int(output[ind][batch_ind][-2].data.cpu().numpy()[0]), int(output[ind][batch_ind][-1].data.cpu().numpy()[0]))
                    sys.stdout.write('[%s] ' % (fj_str) )
                sys.stdout.write('%s ' % (d[val_data[ind,i+batch_ind]]))
            sys.stdout.write('\n')

    print("Done parsing")

    
if __name__ == '__main__':
    main(sys.argv[1:])

