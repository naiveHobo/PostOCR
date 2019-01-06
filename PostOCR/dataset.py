import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):

    def __init__(self, raw_path, gs_path, seq_length=100, batch_size=64, shuffle=True):
        'Initialization'
        self.raw_text = self._load_text(raw_path)
        self.gs_text = self._load_text(gs_path)
        self.chars = sorted(list(set(self.raw_text).union(set(self.gs_text))))
        self.char_to_idx = dict((c, i) for i, c in enumerate(self.chars))
        self.idx_to_char = dict((i, c) for i, c in enumerate(self.chars))
        self.seq_length = seq_length
        self.char_size = len(self.gs_text)
        self.vocab_size = len(self.chars)
        print("Total Characters: {}".format(self.char_size))
        print("Total Vocab: {}".format(self.vocab_size))
        self.total_examples = self.char_size - self.seq_length
        self.indexes = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    @staticmethod
    def _load_text(path):
        with open(path) as f:
            text = f.read()
            text.lower()
        return text

    def __len__(self):
        return int(np.floor(self.total_examples / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        data_x = []
        data_y = []
        for i in indexes:
            seq_in = self.raw_text[i:i + self.seq_length]
            seq_out = self.raw_text[i + self.seq_length]
            data_x.append([self.char_to_idx[char] for char in seq_in])
            data_y.append(self.char_to_idx[seq_out])

        X = np.reshape(data_x, (self.batch_size, self.seq_length, 1))
        X = X / float(self.vocab_size)
        y = keras.utils.to_categorical(data_y, num_classes=self.vocab_size)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.total_examples)
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == '__main__':
    data = DataGenerator(raw_path='../HoboNet/data/raw_text.txt',
                         gs_path='../HoboNet/data/gold_standard.txt')
    print("Total examples: {}".format(len(data)))
    print("\nExample:\n")
    print(data[0])
    print("\nX shape: {}".format(data[0][0].shape))
    print("y shape: {}".format(data[0][1].shape))
