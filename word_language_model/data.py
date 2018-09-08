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
def batchify(data, bsz, cuda=True):

    for sent_len in data.keys():
        # Work out how cleanly we can divide the dataset into bsz parts.
        data_width = data[sent_len].size(0) // (sent_len * bsz)

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        if data_width > 0:
            data[sent_len] = data[sent_len].narrow(0, 0, data_width * sent_len * bsz)

        if data[sent_len].size(0) > bsz and data[sent_len].size(0) % bsz != 0:
            print("Error: Data is not a multiple of batch size")

        # Evenly divide the data across the bsz batches.
        data[sent_len] = data[sent_len].view(-1, sent_len).t().contiguous()
        if cuda:
            data[sent_len] = data[sent_len].cuda()
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
## Modified later (8/14/18) to use 0:n-1th words to predict 1:nth words
def get_batch(source, i, bsz, prefix_len, evaluation=False):
    if i+bsz > source.shape[1]:
        end = source.shape[1]
    else:
        end = i+bsz

    data = Variable(source[:prefix_len,i:end], volatile=evaluation)
    target = Variable(source[1:prefix_len+1,i:end])
    return data, target

def add_unk(data, corpus):
    import random
    dictionary = corpus.dictionary
    unk_ind = dictionary.word2idx['unk']
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if len((corpus.get_num_singletons() == data[i,j].data.cpu()[0]).nonzero()) > 0:
                # this data point is a singleton, with a coin flip replace it with unk
                if random.random() > 0.5:
                    data[i,j] = 0


    return data

# We need the corpus (its mappings specifically) to do any new parsing, but
# sometimes we want to use our model to parse a file outside our original corpus.
# This function reads a file formatted like our corpus, and converts it into
# a 'dataset' object like the corpus.{train|valid|test} objects.
def read_file_outside_corpus(fn, corpus):
    with open(fn, 'r') as f:
        tokens = []
        sents = []
        for line in f:
            words = line.split()
            sents.append(words)
            ids = torch.LongTensor(len(words),1)
            
            token = 0
            for word in words:
                if word not in corpus.dictionary.word2idx:
                    ids[token,0] = corpus.dictionary.word2idx['unk']
                else:
                    ids[token,0] = corpus.dictionary.word2idx[word]
                token += 1

            tokens.append(ids)

    return tokens, sents


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.add_word('unk')

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, max_seq_len):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), max_seq_len)
        self.train_hist = None
        self.train_singletons = None
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), max_seq_len)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), max_seq_len)

    def get_num_singletons(self):
        if self.train_hist is None:
            all_tokens = []
            for subset in self.train:
                all_tokens.extend(self.train[subset].view(1,-1).squeeze().cpu().tolist())

            token_tensor = torch.FloatTensor(all_tokens)
            self.train_hist = torch.histc(token_tensor, bins=len(self.dictionary.idx2word), min=0, max=0)
            self.train_singletons = (self.train_hist==1).nonzero().int() + 1
        return self.train_singletons

    def tokenize(self, path, max_seq_len=100):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary if the sequence is shorter than max_len
        with open(path, 'r') as f:
            tokens = {}
            for line in f:
                words = line.split() #+ ['<eos>']

                if len(words) > max_seq_len or len(words) < 2:
                    continue
                else:
                    if not len(words) in tokens:
                        tokens[len(words)] = 0
                    tokens[len(words)] += len(words)

                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        token_inds = {key:0 for key in tokens.keys()}
        with open(path, 'r') as f:
            ids = {} #torch.LongTensor(tokens)
            for line in f:
                words = line.split() #[:seq_len] # + ['<eos>']
                if len(words) > max_seq_len or len(words) < 2:
                    # ignore lines (sentences) with < 3 tokens
                    continue

                if not len(words) in ids:
                    ids[len(words)] = torch.LongTensor( tokens[len(words)] )

                for word in words:
                    ids[len(words)][token_inds[len(words)]] = self.dictionary.word2idx[word]
                    token_inds[len(words)] += 1

        return ids
