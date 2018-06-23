
from os.path import join
import sys
from  data import Corpus, batchify, get_batch, read_file_outside_corpus
import torch
from torch.autograd import Variable
import argparse

def main(args):
    parser = argparse.ArgumentParser(description='Neural left corner parser')
    parser.add_argument('--prefix-len', type=int, default=3,
                        help='prefix length of sentences used to train the model')
    parser.add_argument('--data', type=str, default='./data/simple',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, required=True,
                        help='Saved model file to use for parsing')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--test', action='store_true', help="Parse the test set (system parses                          validation set by default")
    parser.add_argument('--input', type=str, required=False,
                        help='If this argument is provided, the script will parse the file argument rather than the dev directory of the data argument.')

    args = parser.parse_args(args)

    eval_batch_size = 1

    corpus = Corpus(args.data, seq_len=args.prefix_len)
    d = corpus.dictionary.idx2word

    if args.input is None:
        if args.test:
            input_file = join(args.data, 'test.txt')
        else:
            input_file = join(args.data, 'valid.txt')
    else:
        input_file = args.input

    val_data = read_file_outside_corpus(input_file, corpus)

    with open(args.model, 'rb') as f:
        model = torch.load(f)
        model.parsing = True
        if args.cuda:
            model.cuda()
   
    model.eval()
    ntypes = len(corpus.dictionary)

    for i in range(0, len(val_data)):
        hidden = model.init_hidden(eval_batch_size)
        # data, targets = get_batch(val_data, i, eval_batch_size, prefix_len=prefix_len, evaluation=True)
        data = Variable(val_data[i])
        if args.cuda:
            data = data.cuda()
        output, hidden = model(data, hidden)
        for batch_ind in range(eval_batch_size):
            for ind in range(len(data)):
                fj_str = "%d/%d" % (int(output[ind][batch_ind][-2].data.cpu().numpy()[0]), int(output[ind][batch_ind][-1].data.cpu().numpy()[0]))
                sys.stdout.write('[%s] ' % (fj_str) )
                sys.stdout.write('%s ' % (d[val_data[i][ind].numpy()[0]]))
            sys.stdout.write('\n')

    print("Done parsing")

    
if __name__ == '__main__':
    main(sys.argv[1:])

