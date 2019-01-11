# -*- coding: utf-8 -*-
"""
Takes in 2 files as arguments. The first is a text file that contains a list of training file paths,
with each training file path on a separate line, and the second file is the test data.
The program reads in the files, builds a seq2seq neural model based on Tensorflow, dynamically trains on the training data
in batches for a number of epochs (as determined by the EPOCHS variable), and tests the trained model on the test data.

*All data is expected to have been preprocessed by align.py*
"""

import numpy as np
import tensorflow as tf
import time
import json
import os
import sys

from model import build_bidirectional_graph_with_dynamic_rnn
from config import *


def get_data(file_name):
    max_len = 0
    all_data = ""
    raw_x = []
    raw_y = []

    with open(file_name, 'r') as f:
        count = 1
        for line in f.readlines():
            line = line.strip() + '\n'
            if count % 3 == 1:
                raw_y.append(list(line))
            elif count % 3 == 2:
                raw_x.append(list(line))
            count += 1
            if len(line) > max_len:
                max_len = len(line)
            all_data += line
    print(file_name, max_len)
    return raw_x, raw_y, all_data, max_len


def make_train_dict(tr_data):
    vocab = set(tr_data)
    idx_to_vocab = dict()
    idx_to_vocab[0] = "<PAD>"
    idx_to_vocab[1] = "<UNK>"
    count = 2
    for char in vocab:
        idx_to_vocab[count] = char
        count += 1

    vocab_size = len(idx_to_vocab)
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

    return idx_to_vocab, vocab_to_idx, vocab_size


assert os.path.exists(TRAIN_FILE), "ERROR: Cannot find {}".format(TRAIN_FILE)

tr_raw_x = []
tr_raw_y = []
all_tr_data = ""
tr_max_len = 0

with open(TRAIN_FILE, 'r') as f_names:
    for name in f_names.readlines():
        print("Getting data from " + name)
        name = name.strip()
        r_x, r_y, data, mx_len = get_data(name)
        tr_raw_x += r_x
        tr_raw_y += r_y
        all_tr_data += data
        if tr_max_len < mx_len:
            tr_max_len = mx_len


idx_vocab, vocab_idx, tr_vocab_size = make_train_dict(all_tr_data)
vocab_tuple = (idx_vocab, vocab_idx, tr_vocab_size)

# All input arrays are padded to get uniform input
MAX_LEN = tr_max_len
print("\n+++++++++++++++")
print("Max sequence length: {}".format(MAX_LEN))
print("+++++++++++++++\n")


# Character to integer conversion and padding
tr_X_data = [[vocab_idx[c] for c in arr] for arr in tr_raw_x]
tr_Y_data = [[vocab_idx[c] for c in arr] for arr in tr_raw_y]

tr_X_data = np.array([np.pad(line, (0, MAX_LEN-len(line)), 'constant', constant_values=0) for line in tr_X_data])
tr_Y_data = np.array([np.pad(line, (0, MAX_LEN-len(line)+EDIT_SPACE), 'constant', constant_values=0) for line in tr_Y_data])

# Save dictionaries for future use
with open(MODEL_PATH + "-vocab-dictionaries", 'w') as jsonfile:
    json.dump(vocab_tuple, jsonfile)

t = time.time()
graph = build_model(num_classes=tr_vocab_size)

writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

TRAIN_STEPS_PER_EPOCH = len(tr_X_data) // BATCH_SIZE

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    global_step = 0

    sess.run(graph['data_iter'].initializer,
             feed_dict={graph['x']: tr_X_data,
                        graph['y']: tr_Y_data,
                        graph['batch_size']: BATCH_SIZE})

    for i in range(NUM_EPOCHS):
        steps = 0
        tot_loss = 0
        training_state = None

        print("\nEPOCH: {}".format(i))

        for _ in range(TRAIN_STEPS_PER_EPOCH):
            loss_value, training_state, _, summary = sess.run([graph['total_loss'],
                                                               graph['final_state'],
                                                               graph['train_step'],
                                                               graph['train_summary']])

            if steps % 10 == 0:
                writer.add_summary(summary, global_step)

            print("Step: {} \t Loss: {}".format(global_step, loss_value))

            tot_loss += loss_value
            steps += 1
            global_step += 1

            if global_step % SAVE_CHECKPOINT_STEP == 0:
                print("\nSaving checkpoint...\n")
                graph['saver'].save(sess, os.path.join(CHECKPOINT_DIR,
                                                       "hobonet-{}-{:.4f}.ckpt".format(global_step, tot_loss / steps)),
                                    global_step=global_step)

        print("\nAverage training loss: {}\n".format(tot_loss / steps))

    graph['saver'].save(sess, MODEL_PATH)

print("\nTraining was completed!\n")
