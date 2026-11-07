"""Model neural network sederhana untuk prediksi probabilitas gagal bayar."""
import torch
import torch.nn as nn


class CreditMLP(nn.Module):
    """
    Input  : fitur numerik (sudah distandarisasi) + fitur kategorikal (indeks)
    Output : satu logit -> sigmoid -> probabilitas gagal bayar (0..1)
    """

    def __init__(self, n_numeric, cardinalities, emb_dim=4):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, emb_dim) for card in cardinalities]
        )

        in_dim = n_numeric + emb_dim * len(cardinalities)

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x_num, x_cat):
        parts = [x_num]
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(x_cat[:, i]))
        x = torch.cat(parts, dim=1)
        return self.net(x).squeeze(1)
