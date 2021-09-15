import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import predictor.constants as c
from predictor.data import DNASeqDataset, collate_fn, load_data
from predictor.evaluate import evaluate
from predictor.model import BiLSTM
from predictor.preprocessor import DataPreprocessor, NegativeSampleGenerator
from predictor.train import train
from predictor.utils import argument_parser, set_seeds
from predictor.vocabulary import Vocab

if __name__ == "__main__":
    set_seeds()
    args = argument_parser()

    data = load_data(args.data_path)
    vocab = Vocab(data)

    neg_sample_gen = NegativeSampleGenerator(vocab)
    preprocessor = DataPreprocessor(vocab, neg_sample_gen)

    if args.mode == "train":
        train_data, test_data = train_test_split(data, test_size=0.1)

        train_seqs, train_labels = preprocessor.process_train(train_data)
        valid_seqs, valid_labels = preprocessor.process_train(test_data)
        print(f"Train size: {len(train_seqs)}")
        print(f"Validation size: {len(valid_seqs)}")

        train_dataset = DNASeqDataset(train_seqs, train_labels)
        valid_dataset = DNASeqDataset(valid_seqs, valid_labels)
        train_iter = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        valid_iter = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
    elif args.mode == "eval":
        test_seqs, test_labels = preprocessor.process_inference(data)
        print(f"Test size: {len(test_seqs)}")
        test_dataset = DNASeqDataset(test_seqs, test_labels)
        test_iter = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
    else:
        raise Exception("Mode not defined")

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
            args.model_path,
        )
        acc = evaluate(model, valid_iter, args.model_path)
        print(f"Validation accuracy: {acc}%")
    else:
        acc = evaluate(model, test_iter, args.model_path)
        print(f"Test accuracy: {acc}%")
