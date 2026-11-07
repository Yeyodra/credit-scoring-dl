"""Model MLP sederhana untuk prediksi probabilitas gagal bayar."""
import torch.nn as nn


class CreditMLP(nn.Module):
    """MLP polos: 23 fitur terstandardisasi -> 64 -> 32 -> 1 (logit).

    Tanpa embedding / one-hot. Fitur kategorikal (SEX, EDUCATION, MARRIAGE)
    cukup dipakai sebagai angka lalu distandarisasi, sama seperti fitur lain.
    Ini jauh lebih sederhana namun tetap memberi hasil yang memadai.
    """

    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
