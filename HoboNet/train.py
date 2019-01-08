import os
import h5py
import keras
import random
import argparse

from model import get_model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="./data/data.h5", help='Path to HDF5 file containg data')
parser.add_argument('--train_dir', type=str, default="./checkpoints/", help='Path to Weights checkpoint')

args = parser.parse_args()

assert os.path.exists(args.dataset), "Prepped data file is not available"

if not os.path.exists(args.train_dir):
    os.mkdir(args.train_dir)

if not os.path.exists('./model/'):
    os.mkdir('./model/')


hf = h5py.File(args.dataset, 'r')

num_epochs = 50
batch_size = 64
enc_seq_length = 35 
dec_seq_length = 35
vocab_size = 113

inp_x = hf['train/raw_sent_mat'][:, :enc_seq_length]
inp_cond_x = hf['train/gs_sent_mat'][:, :dec_seq_length]
out_y = hf['train/gs_sent_mat'][:, 1:dec_seq_length + 1]

val_x = hf['val/raw_sent_mat'][:, :enc_seq_length]
val_cond_x = hf['val/gs_sent_mat'][:, :dec_seq_length]
val_y = hf['val/gs_sent_mat'][:, 1:dec_seq_length + 1]

tr_data = list(range(inp_x.shape[0]))
val_data = list(range(val_x.shape[0]))


def load_train_data(batch_size=64):
    while True:
        random.shuffle(tr_data)
        for i in range(0, len(tr_data) - batch_size, batch_size):
            idxs = tr_data[i:i + batch_size]
            yield [inp_x[idxs], inp_cond_x[idxs]], keras.utils.to_categorical(out_y[idxs], num_classes=vocab_size)


def load_val_data(batch_size=64):
    while True:
        for i in range(0, len(val_data) - batch_size, batch_size):
            idxs = val_data[i:i + batch_size]
            yield [val_x[idxs], val_cond_x[idxs]], keras.utils.to_categorical(val_y[idxs], num_classes=vocab_size)


tr_gen = load_train_data(batch_size=batch_size)
val_gen = load_val_data(batch_size=batch_size)

m = get_model()
best_val_loss = None


for ep in range(num_epochs):
    print("Epoch: {}".format(ep))
    m.fit_generator(tr_gen, steps_per_epoch=100, epochs=1)
    loss = m.evaluate_generator(val_gen, steps=len(val_data)//batch_size, verbose=True)
    if not best_val_loss:
        best_val_loss = loss
        continue
    if loss < best_val_loss:
        print("\n===========================\n")
        print("Saving model, epoch: {}, val loss: {}".format(ep, loss))
        print("\n===========================\n")
        m.save_weights(os.path.join(args.train_dir, 'model-{}-{:.4f}.h5'.format(ep, loss)))
        best_val_loss = loss

m.save_weights('./model/model.h5')

print("Training is finished")
