import h5py
import argparse
import json
import numpy as np
import math

from model import get_model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="./data/prepped.h5", help='Path to HDF5 file containing data')
parser.add_argument('--weights_path', type=str, default="./weights/model.h5", help='Path to Weights checkpoint')

args = parser.parse_args()

hf = h5py.File(args.dataset, 'r')
m = get_model()
m.load_weights(args.weights_path)

vocab = json.loads(hf['vocab'].value)


def greedy_decoder(data):
    return [np.argmax(d) for d in data]


def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -math.log(row[j])]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:k]
    return sequences


def predict(sentence, decoder_fn=greedy_decoder):
    sentences = []
    for i in range(0, len(sentence), 35):
        sentences.append(sentence[i:i+35].lower())

    ret = ""
    for sent in sentences:
        chars = []

        for c in sent:
            if c in vocab['char2idx']:
                chars.append(vocab['char2idx'][c])
            else:
                chars.append(vocab['char2idx']['#'])


        m_input = [np.zeros((1, 35)), np.zeros((1, 35))]
        for i, c in enumerate(chars):
            m_input[0][0, i] = c

        for c_i in range(1, 35):
            out = m.predict(m_input)
            out_c_i = decoder_fn(out[0][c_i-1])

            if out_c_i == 0:
                continue

            ret += vocab['idx2char'][str(out_c_i)]
            m_input[1][0, c_i] = out_c_i

    return ret


while True:
    print("Enter a sentence: ")
    sent = input()
    print(predict(sent))
    print("===============")
