import os

import numpy as np
import torch
from tqdm import tqdm

import constants as c
from utils import plot, save_checkpoint


def train(
    model,
    optimizer,
    criterion,
    train_iter,
    valid_iter,
    num_epochs,
    eval_every,
    output_path,
):
    train_loss_list = []
    valid_loss_list = []
    valid_running_loss = 0
    train_running_loss = 0
    best_valid_loss = float("inf")
    step = 0
    model_path = os.path.join(output_path, c.MODEL_NAME)
    fig_path = os.path.join(output_path, c.FIG_NAME)
    model.train()
    for e in range(num_epochs):
        print(f"Epoch number {e}")
        for seqs, labels, lengths in tqdm(train_iter):
            output = model(seqs, lengths)
            loss = criterion(output, labels)
            train_loss_list.append(loss.item())
            train_running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for seqs, labels, lengths in valid_iter:
                        output = model(seqs, lengths)
                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()
                        valid_loss_list.append(loss.item())

                model.train()

                avg_train_loss = train_running_loss / len(train_iter)
                avg_valid_loss = valid_running_loss / len(valid_iter)
                print(f"Train loss: {avg_train_loss}")
                print(f"Validation loss: {avg_valid_loss}")
                train_running_loss = 0
                valid_running_loss = 0
                if best_valid_loss >= avg_valid_loss:
                    best_valid_loss = avg_valid_loss
                    save_checkpoint(model_path, model, optimizer)

    plot(step, train_loss_list, valid_loss_list)
