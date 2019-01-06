import os
import gzip
import pickle as pkl
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard

from .dataset import DataGenerator


class HoboOCR:

    def __init__(self):
        self.model = None
        self.char_to_idx = None
        self.idx_to_char = None

    def build_model(self, input_shape, out_size, num_hidden=256, dropout=0.2, load_weights=None):
        if self.model:
            return self.model

        model = Sequential()
        model.add(LSTM(num_hidden, input_shape=(input_shape[0], input_shape[1]), return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(num_hidden, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(num_hidden))
        model.add(Dropout(dropout))
        # model.add(LSTM(256))
        # model.add(Dropout(0.2))
        model.add(Dense(out_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        if load_weights and os.path.exists(load_weights):
            model.load_weights(load_weights)

        self.model = model
        return model

    def create_char_to_idx_map(self, data_path=None):
        if os.path.exists(data_path):
            with gzip.open(data_path, 'rb') as f:
                data = pkl.load(f)
            self.char_to_idx = data['char_to_idx']
            self.idx_to_char = data['idx_to_char']
        else:
            data = {'char_to_idx': self.char_to_idx, 'idx_to_char': self.idx_to_char}
            with gzip.open(data_path, 'wb') as f:
                pkl.dump(data, f)

    def train(self, data, train_dir, log_dir, num_epochs=50, batch_size=64, val_split=0.125):
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq=100)
        model_checkpoints = ModelCheckpoint(os.path.join(train_dir, "model_") +
                                            ".{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5",
                                            monitor='val_loss', verbose=0, save_best_only=True,
                                            save_weights_only=False, mode='auto')
        callbacks = [tensorboard, model_checkpoints]

        self.model.fit_generator(generator=data,
                                 epochs=num_epochs,
                                 steps_per_epoch=10000,
                                 verbose=True,
                                 callbacks=callbacks,
                                 use_multiprocessing=True,
                                 workers=6)

    def detect_errors(self):
        pass

    def correct_errors(self):
        pass


if __name__ == '__main__':
    hobo = HoboOCR()
    gen = DataGenerator(raw_path='../HoboNet/data/raw_text.txt',
                        gs_path='../HoboNet/data/gold_standard.txt')
    hobo.build_model(input_shape=(gen.seq_length, 1),
                     out_size=gen.vocab_size)
    hobo.train(gen, train_dir='../HoboNet/checkpoints/', log_dir='../HoboNet/logs/')
