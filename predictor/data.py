import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class DNASeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    seqs, labels = zip(*batch)
    seqs = [torch.tensor(seq, dtype=torch.int) for seq in seqs]
    labels = torch.tensor(labels, dtype=torch.float)
    padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    lengths = [len(seq) for seq in seqs]
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded_seqs, labels, lengths


def load_data(path):
    with open(path, "r") as f:
        data = f.read()
        data = data.strip().split("\n")

    print(f"Number of data points: {len(data)}")
    return data
