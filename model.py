import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bi_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, x_len):
        emb = self.embedding(x)
        packed_emb = pack_padded_sequence(
            emb, x_len, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bi_lstm(packed_emb)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        forward_out = out[torch.arange(len(out)), x_len - 1, : self.hidden_dim]
        backward_out = out[:, 0, self.hidden_dim :]
        cat_out = torch.cat([forward_out, backward_out], 1)

        cat_out = self.dropout(cat_out)

        final_out = self.linear(cat_out)
        final_out = final_out.squeeze(1)
        return torch.sigmoid(final_out)
