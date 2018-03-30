import os
import torch
from torch.autograd import Variable

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def batchify(data, bsz, prefix_len):
    # Work out how cleanly we can divide the dataset into bsz parts.
    data_width = data.size(0) // (prefix_len * bsz)

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, data_width * prefix_len * bsz)
    if data.size(0) % prefix_len != 0:
        print("Error: Data is not a multiple of 3")
    if data.size(0) % bsz != 0:
        print("Error: Data is not a multiple of batch size")

    # Evenly divide the data across the bsz batches.
    data = data.view(-1, prefix_len).t().contiguous()
    data = data.cuda()
    return data

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
## Modified by TM to take the next seq_len-1 words and use it to predict the seq_len_th word
def get_batch(source, i, bsz, prefix_len=3, evaluation=False):
    data = Variable(source[:prefix_len-1,i:i+bsz], volatile=evaluation)
    target = Variable(source[prefix_len-1,i:i+bsz].view(-1))
    return data, target

# def get_batch(source, i, bsz, evaluation=False):
#     seq_len = args.prefix_len # min(args.bptt, len(source) - 1 - i)
#     data = Variable(source[:seq_len-1,i:i+bsz], volatile=evaluation)
#     target = Variable(source[seq_len-1,i:i+bsz].view(-1))
#     return data, target

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, seq_len):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), seq_len)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), seq_len)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), seq_len)

    def tokenize(self, path, seq_len):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split()[:seq_len] #+ ['<eos>']
                if len(words) != seq_len:
                    # ignore lines (sentences) with < 3 tokens
                    continue

                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split()[:seq_len] # + ['<eos>']
                if len(words) != seq_len:
                    # ignore lines (sentences) with < 3 tokens
                    continue

                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
