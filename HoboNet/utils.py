import numpy as np
import gzip
import pickle as pkl


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.idx = 1  # id 0 is reserved for padding
        self.add_char('#')

    def add_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.idx
            self.idx2char[self.idx] = char
            self.idx += 1

    def add_chars(self, chars):
        for c in chars:
            self.add_char(c)

    def add_sentence(self, sentence):
        chars = sentence.lower()
        for c in chars:
            self.add_char(c)

    def __call__(self, char):
        if char not in self.char2idx:
            return self.char2idx['#']
        return self.char2idx[char]

    def __len__(self):
        return len(self.char2idx)

    def get_dicts(self):
        dicts = {
            "idx": self.idx,
            "idx2char": self.idx2char,
            "char2idx": self.char2idx
        }
        return dicts

    def save(self, save_as):
        data = {
            'idx': self.idx,
            'char2idx': self.char2idx,
            'idx2char': self.idx2char
        }
        with gzip.open(save_as, 'wb') as f:
            pkl.dump(data, f)

    def load(self, path):
        with gzip.open(path, 'rb') as f:
            data = pkl.load(f)
        self.idx = data['idx']
        self.idx2char = data['idx2char']
        self.char2idx = data['char2idx']


def get_sentences_matrix(sentences, vocab, max_sentence_len=None, padding='right'):
    if max_sentence_len is None:
        max_sentence_len = max([len(s) for s in sentences])
    sentences_mat = np.zeros((len(sentences), max_sentence_len)).astype('int64')
    for i, sen in enumerate(sentences):
        ids = [vocab(w) for w in sen]
        if padding == 'right':
            ids = ids[:max_sentence_len]
        else:
            ids = ids[-1*max_sentence_len:]

        if len(ids) < max_sentence_len:
            if padding == 'right':
                ids = ids + [0]*(max_sentence_len - len(ids))
            else:
                ids = [0]*(max_sentence_len - len(ids)) + ids
        sentences_mat[i] = np.array(ids)
    return sentences_mat
