
from os.path import join
import sys
from  data import Corpus, batchify, get_batch, read_file_outside_corpus
import torch
from torch.autograd import Variable
import argparse

def main(args):
    parser = argparse.ArgumentParser(description='Neural left corner parser')
    parser.add_argument('--model', type=str, required=True,
                        help='Saved model file to use for parsing')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--test', action='store_true', help="Parse the test set (system parses                          validation set by default)")
    parser.add_argument('--debug', action='store_true', help="Print out debugging information during parsing")
    parser.add_argument('--input', type=str, required=False,
                        help='If this argument is provided, the script will parse the file argument rather than the dev directory of the data argument.')

    args = parser.parse_args(args)

    eval_batch_size = 1
    device = torch.device("cuda" if args.cuda else "cpu")

    with open(args.model, 'rb') as f:
        model = torch.load(f)
        model.parsing = True
        model = model.to(device)
        model.debug = args.debug

    corpus = model.corpus
    d = corpus.dictionary.idx2word

    if args.input is None:
        if args.test:
            input_file = join(args.data, 'test.txt')
        else:
            input_file = join(args.data, 'valid.txt')
    else:
        input_file = args.input

    val_data, val_sents = read_file_outside_corpus(input_file, corpus)

    model.eval()

    for i in range(0, len(val_data)):
        hidden = model.init_hidden(eval_batch_size)
        # data, targets = get_batch(val_data, i, eval_batch_size, prefix_len=prefix_len, evaluation=True)
        data = Variable(val_data[i])
        data = data.to(device)
        output, hidden = model(data, hidden)
        for batch_ind in range(eval_batch_size):
            for ind in range(len(data)):
                fj_str = "%d/%d" % (int(output[ind][batch_ind][-2].data.cpu().item()), int(output[ind][batch_ind][-1].data.cpu().item()))
                sys.stdout.write('[%s] ' % (fj_str) )
                sys.stdout.write('%s ' % (val_sents[i][ind]))
            sys.stdout.write('\n')

    sys.stderr.write("Done parsing\n")

    
if __name__ == '__main__':
    main(sys.argv[1:])

