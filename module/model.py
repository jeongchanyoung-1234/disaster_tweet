import torch.nn as nn

class DisasterClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_dim,
                 num_layers,
                 hidden_size,
                 dropout,
                 n_classes):
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_classes = n_classes

        super().__init__()

        self.Embedding = nn.Embedding(
            num_embeddings=self.input_size,
            embedding_dim=self.embedding_dim,
        )
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.generator = nn.Linear(
            in_features=hidden_size * 2,
            out_features=self.n_classes
        )
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.Embedding(x)
        h, _ = self.rnn(x)
        x = self.generator(h[:, -1])
        y = self.activation(x)

        return y