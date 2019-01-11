import os
import json
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

from config import *


parser = ArgumentParser()

parser.add_argument('--input_file',
                    help='path to text file containing data to run HoboNet on',
                    required=True)

parser.add_argument('--output_file',
                    help='name with which HoboNet output should be saved',
                    default="output.txt")

parser.add_argument('--model_path',
                    help='path to directory containing saved HoboNet model',
                    default="./model/")


args = parser.parse_args()

vocab_idx_file = os.path.join(args.model_path, "hobonet-vocab-dictionaries")


def get_data(file_name):
    with open(file_name, 'r') as f:
        raw_x = []
        for line in f.readlines():
            line = line.strip() + '\n'
            if line:
                raw_x.append(list(line))
        print("X Data Length:", len(raw_x))
    return raw_x


with open(vocab_idx_file) as vocab_file:
    vocab_tuple = json.load(vocab_file)
    idx_vocab = vocab_tuple[0]
    vocab_idx = vocab_tuple[1]
    tr_vocab_size = vocab_tuple[2]


raw_x = get_data(args.input_file)

data_x = [[vocab_idx[c] if c in vocab_idx else vocab_idx['<UNK>'] for c in arr] for arr in raw_x]

data_y = np.array([np.pad(line, (0, MAX_SEQ_LENGTH - len(line) + EDIT_SPACE), 'constant', constant_values=0)
                   for line in data_x])
data_x = np.array([np.pad(line, (0, MAX_SEQ_LENGTH - len(line)), 'constant', constant_values=0) for line in data_x])


with tf.Session() as sess:
    checkpoint = tf.train.latest_checkpoint(args.model_path)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint))
    saver.restore(sess, checkpoint)
    g = tf.get_default_graph()

    init_op = tf.global_variables_initializer()
    batch_size = g.get_tensor_by_name("HoboNet/encoder/batch_size:0")

    x = g.get_tensor_by_name("HoboNet/encoder/input_placeholder:0")
    y = g.get_tensor_by_name("HoboNet/encoder/labels_placeholder:0")

    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
    make_iter = g.get_operation_by_name("HoboNet/encoder/MakeIterator")

    sess.run(make_iter, feed_dict={x: data_x,
                                   y: data_y,
                                   batch_size: len(data_x)})

    preds = g.get_tensor_by_name("HoboNet/decoder/preds:0")

    output = sess.run(preds)

    idx = np.where(output == 0)
    new_out = np.delete(output, idx)

    char_func = np.vectorize(lambda t: idx_vocab[str(t)])
    chars = char_func(new_out)
    result = "".join(chars)


with open(args.output_file, 'w') as out:
    out.write(result.strip())
