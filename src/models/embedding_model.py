'''
In the paper they use a CNN over spectrograms, but we can use
an LSTM over the lyrics and classify [artist/genre/decade]. We'll use
the weight matrix from the last layer of the classification as the embedding.
'''

import torch
import torch.nn as nn


class GenreEmbedding_LSTM(nn.Module):
    def __init__(self, vocab_size, lstm_embedding_dim, lstm_hidden_dim, genre_embedding_dim, num_categories):
        super(GenreEmbedding_LSTM, self).__init__()

        self.vocab_size = vocab_size
        self.lstm_embedding_dim = lstm_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.genre_embedding_dim = genre_embedding_dim
        self.num_categories = num_categories

        self.lstm_embedding = nn.Embedding(
            vocab_size, embedding_dim=lstm_embedding_dim)
        self.lstm_encoder = nn.LSTM(
            lstm_embedding_dim, lstm_hidden_dim, batch_first=True)
        self.hidden2genre = nn.Linear(lstm_hidden_dim, genre_embedding_dim)
        self.relu = nn.ReLU()
        self.genre2category = nn.Linear(genre_embedding_dim, num_categories)

    def get_embeddings(self):
        # returns (num_categories, genre_embedding_dim)
        return self.genre2category.weight

    def forward(self, x):
        # x.shape = (batch_size, seq_len)
        x = self.lstm_embedding(x)
        # x.shape = (batch_size, seq_len, lstm_embedding_dim)
        x, _ = self.lstm_encoder(x)
        # x.shape = (batch_size, seq_len, lstm_hidden_dim)
        x = x[:, -1, :]
        # x.shape = (batch_size, lstm_hidden_dim)
        x = self.hidden2genre(x)
        # x.shape = (batch_size, genre_embedding_dim)
        x = self.relu(x)
        # x.shape = (batch_size, genre_embedding_dim)
        x = self.genre2category(x)
        # x.shape = (batch_size, num_categories)
        return x


def train_embedding_model(model, train_loader, epochs, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: {loss.item()}')


def test_embedding_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_hat = model(x)
            _, predicted = torch.argmax(y_hat, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f'Accuracy: {100 * correct / total}')


model = GenreEmbedding_LSTM(vocab_size=100, lstm_embedding_dim=10,
                            lstm_hidden_dim=20, genre_embedding_dim=30, num_categories=5)
print(model.get_embeddings().shape)
