import os

import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import constants as c
from data import DNASeqDataset, collate_fn
from evaluate import evaluate
from model import BiLSTM
from preprocessor import DataPreprocessor, NegativeSampleGenerator
from train import train
from utils import load_data, set_seeds, argument_parser
from vocabulary import Vocab


if __name__ == "__main__":
    set_seeds()
    args = argument_parser()

    data = load_data(args.data_path)
    vocab = Vocab(data)
    train_data, test_data = train_test_split(data, test_size=0.1)

    neg_sample_gen = NegativeSampleGenerator(vocab)
    preprocessor = DataPreprocessor(vocab, neg_sample_gen)
    train_seqs, train_labels = preprocessor.process(train_data)
    valid_seqs, valid_labels = preprocessor.process(test_data)
    print(f"Train size: {len(train_seqs)}")
    print(f"Test size: {len(valid_seqs)}")

    train_dataset = DNASeqDataset(train_seqs, train_labels)
    valid_dataset = DNASeqDataset(valid_seqs, valid_labels)
    train_iter = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_iter = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    model = BiLSTM(vocab.size, args.emb_dim, args.hid_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    if args.mode == "train":
        train(
            model,
            optimizer,
            criterion,
            train_iter,
            valid_iter,
            args.n_epochs,
            len(train_iter) // 2,
            c.PATH,
        )
    acc = evaluate(model, valid_iter, os.path.join(c.PATH, c.MODEL_NAME))
    print(f"Validation accuracy: {acc}%")
