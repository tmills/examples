
import sys
from  data import Corpus, batchify, get_batch
import torch
from torch.autograd import Variable

prefix_len = 3

# def batchify(data, bsz):
#     # Work out how cleanly we can divide the dataset into bsz parts.
#     data_width = data.size(0) // (prefix_len * bsz)

#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.narrow(0, 0, data_width * prefix_len * bsz)
#     if data.size(0) % prefix_len != 0:
#         print("Error: Data is not a multiple of %d" % (prefix_len))
#     if data.size(0) % bsz != 0:
#         print("Error: Data is not a multiple of batch size")

#     # Evenly divide the data across the bsz batches.
#     data = data.view(-1, 3).t().contiguous()
#     data = data.cuda()
#     return data


def main(args):
    req_args = 2
    if len(args) < req_args:
        sys.stderr.write("%d required argument(s): <model file> <corpus>\n" % (req_args))
        sys.exit(-1)
    
    corpus = Corpus(args[1], seq_len=prefix_len)
    d = corpus.dictionary.idx2word
    val_data = batchify(corpus.valid, 1, prefix_len=prefix_len)
    eval_batch_size = 1

    with open(args[0], 'rb') as f:
        model = torch.load(f)
        model.parsing = True
   
    model.eval()
    ntypes = len(corpus.dictionary)

    for i in range(0, val_data.size(1) - 1):
        hidden = model.init_hidden(eval_batch_size)
        data, targets = get_batch(val_data, i, eval_batch_size, prefix_len=prefix_len, evaluation=True)
        output, hidden = model(data, hidden)
        for ind in range(prefix_len):
            if ind+1 < prefix_len:
                fj_str = "%d/%d" % (int(output[ind][-1][-2].data.cpu().numpy()[0]), int(output[ind][-1][-1].data.cpu().numpy()[0]))
                sys.stdout.write('[%s] ' % (fj_str) )
            sys.stdout.write('%s ' % (d[val_data[ind,i]]))
            # sys.stdout.write('%s ' % (  ) )

            # print('%s %s [%s] %s' % (d[val_data[0,i]], d[val_data[1,i]], fj_str, d[val_data[2,i]] ))
        sys.stdout.write('\n')

    print("Done parsing")

    
if __name__ == '__main__':
    main(sys.argv[1:])

