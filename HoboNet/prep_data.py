import h5py
import json
import argparse
import glob
import os

from utils import get_sentences_matrix, Vocabulary


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training data')
parser.add_argument('--val_dir', type=str, required=True, help='Directory containing validation data')
parser.add_argument('--out_file', type=str, default="./data/data.h5", help='Output HDF5 file name')
args = parser.parse_args()

raw_sents_train = []
raw_sents_val = []
gs_sents_train = []
gs_sents_val = []

filenames = glob.glob(os.path.join(args.train_dir, "*.txt"))

print('Training files found: {}'.format(len(filenames)))

for filename in filenames:
    with open(filename) as f:
        lines = f.readlines()
        raw_text = lines[0].replace('[OCR_toInput]', '').lower().strip()
        gs_text = lines[2].replace('[ GS_aligned]', '').lower().strip()
        raw_sents_train.append(raw_text)
        gs_sents_train.append(gs_text)

vocab = Vocabulary()

for sent in raw_sents_train:
    vocab.add_chars(sent)

for sent in gs_sents_train:
    vocab.add_chars(sent)

print("Vocabulary Size: {}".format(len(vocab)))

filenames = glob.glob(os.path.join(args.val_dir, "*.txt"))

print('Validation files found: {}'.format(len(filenames)))

for filename in filenames:
    with open(filename) as f:
        lines = f.readlines()
        raw_text = lines[0].replace('[OCR_toInput]', '').lower().strip()
        gs_text = lines[2].replace('[ GS_aligned]', '').lower().strip()
        raw_sents_val.append(raw_text)
        gs_sents_val.append(gs_text)


raw_sent_mat_train = get_sentences_matrix(raw_sents_train, vocab, max_sentence_len=100)
gs_sent_mat_train = get_sentences_matrix(gs_sents_train, vocab, max_sentence_len=100)
raw_sent_mat_val = get_sentences_matrix(raw_sents_val, vocab, max_sentence_len=100)
gs_sent_mat_val = get_sentences_matrix(gs_sents_val, vocab, max_sentence_len=100)


f = h5py.File(args.out_file, "w")

f.create_dataset("vocab", data=json.dumps(vocab.get_dicts()))

f.create_dataset("train/raw_sents", data=json.dumps(raw_sents_train))
f.create_dataset("train/gs_sents", data=json.dumps(gs_sents_train))

f.create_dataset("val/raw_sents", data=json.dumps(raw_sents_val))
f.create_dataset("val/gs_sents", data=json.dumps(gs_sents_val))

f.create_dataset('train/raw_sent_mat', data=raw_sent_mat_train)
f.create_dataset('train/gs_sent_mat', data=gs_sent_mat_train)

f.create_dataset('val/raw_sent_mat', data=raw_sent_mat_val)
f.create_dataset('val/gs_sent_mat', data=gs_sent_mat_val)

f.close()

print("Data successfully prepped for training!")
