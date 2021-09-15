import csv
import random
from typing import Dict, List

import torch

import constants as c
from vocabulary import Vocab


class NegativeSampleGenerator:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def generate(self, input_sample: List[int]):
        rnd = random.random()
        if rnd < 0.25:
            return self.reverse(input_sample)
        elif rnd < 0.5:
            return self.change_codes(input_sample)
        elif rnd < 0.75:
            return self.add_a_block(input_sample)
        else:
            return self.remove_a_block(input_sample)

    def reverse(self, input_sample: List[int]):
        return input_sample[::-1], len(input_sample)

    def change_codes(self, input_sample: List[int], percentage=0.3):
        n_codes_to_change = int(percentage * len(input_sample))
        indexes_to_change = random.sample(range(len(input_sample)), n_codes_to_change)
        for i in indexes_to_change:
            new_vocab = dict(self.vocab.index2word)
            new_vocab.pop(input_sample[i])
            input_sample[i] = random.sample(new_vocab.keys(), 1)[0]
        return input_sample, len(input_sample)

    def add_a_block(self, input_sample: List[int], size_ratio=0.2):
        size = int(len(input_sample) * size_ratio)
        all_indexes = list(self.vocab.index2word.keys())
        sampled_indexes = random.choices(all_indexes, k=size)
        add_index = random.randint(0, len(input_sample))
        input_sample[add_index:add_index] = sampled_indexes
        return input_sample, len(input_sample)

    def remove_a_block(self, input_sample: List[int], size_ratio=0.2):
        size = int(len(input_sample) * size_ratio)
        remove_index = random.randint(0, len(input_sample) - size)
        del input_sample[remove_index : remove_index + size]
        return input_sample, len(input_sample)


class DataPreprocessor:
    def __init__(self, vocab: Vocab, neg_sample_gen: NegativeSampleGenerator):
        self.vocab = vocab
        self.neg_sample_gen = neg_sample_gen

    def process_train(self, data: List[str]):
        pos_seqs = set()
        neg_seqs = set()
        labels = []
        for example in data:
            pos_seq = []
            for char in example:
                pos_seq.append(self.vocab.word2index[char])
            pos_seqs.add(tuple(pos_seq))

        for pos_seq in pos_seqs:
            while True:
                neg_seq, neg_len = self.neg_sample_gen.generate(list(pos_seq))
                neg_seq = tuple(neg_seq)
                if neg_seq not in neg_seqs and neg_seq not in pos_seqs:
                    break

            neg_seqs.add(neg_seq)

        seqs = list(pos_seqs) + list(neg_seqs)
        labels = [c.POSITIVE] * len(pos_seqs) + [c.NEGATIVE] * len(neg_seqs)
        return seqs, labels

    def process_inference(self, data: List[str]):
        seqs = []
        labels = []
        for example in data:
            seq, label = example.split(",")
            indexed_seq = []
            for char in seq:
                indexed_seq.append(self.vocab.word2index[char])
            seqs.append(indexed_seq)
            labels.append(int(label))
        return seqs, labels
