import torch

from utils import load_model


def evaluate(model, valid_iter, path):
    load_model(path, model)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for seqs, labels, lengths in valid_iter:
            outputs = model(seqs, lengths)
            outputs = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (labels == outputs).sum().item()
    return round(correct / total * 100, 2)
