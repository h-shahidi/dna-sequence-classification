import argparse
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_checkpoint(path, model, optimizer):
    state_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state_dict, path)
    print(f"Model saved to {path}")


def load_model(path, model):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict["model_state_dict"])


def set_seeds():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


def moving_average(x, window):
    x = np.array(x)
    return np.convolve(x, np.ones(window), "valid") / window


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--data_path", type=str, default="train.csv")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--model_path", type=str, default="./model.pt")
    return parser.parse_args()


def plot(n_steps: int, train_loss_list: List[float], valid_loss_list: List[float]):
    train_loss_list = moving_average(train_loss_list, 20)
    valid_loss_list = moving_average(valid_loss_list, 20)
    plt.plot(
        np.linspace(0, n_steps, len(train_loss_list)),
        train_loss_list,
        "r",
        label="train",
    )
    plt.plot(
        np.linspace(0, n_steps, len(valid_loss_list)),
        valid_loss_list,
        "b",
        label="validation",
    )
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("loss.png")
