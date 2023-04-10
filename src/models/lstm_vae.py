import torch 
import torch.nn as nn

class LSTM_VAE(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_size, hidden_size, latent_size, genre_embed_size):
        super(LSTM_VAE, self).__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.genre_embed_size = genre_embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        # Encoder
        self.encoder = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(self.hidden_size, self.latent_size)
        self.fc_logvar = nn.Linear(self.hidden_size, self.latent_size)

        # Decoder
        self.fc_z = nn.Linear(self.latent_size + self.genre_embed_size, self.hidden_size)
        self.decoder = nn.LSTM(self.latent_size + self.genre_embed_size, self.hidden_size, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

    def encode(self, x):
        x = self.embedding(x)
        _, (h, _) = self.encoder(x)
        h = h.view(-1, self.hidden_size)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, genre):
        z = torch.cat((z, genre), dim=1)
        z = self.fc_z(z)
        z = z.view(-1, 1, self.hidden_size)
        z = z.repeat(1, self.seq_len, 1)
        op, _ = self.decoder(z)
        op = self.fc_out(op)
        return op
    
    def forward(self, x, genre):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        op = self.decode(z, genre)
        return op, mu, logvar
    
    def sample(self, genre):
        z = torch.randn(1, self.latent_size)
        z = self.decode(z, genre)
        return z

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
    def reconstruction_loss(self, op, x):
        return nn.CrossEntropyLoss()(op, x)

    def vae_loss(self, op, x, mu, logvar):
        recon_loss = self.reconstruction_loss(op, x)
        kl_loss = self.kl_divergence(mu, logvar)
        return recon_loss + kl_loss
    

def train(model, train_loader, epochs, lr=0.001):
    model.train()
    # train_loader is a torch DataLoader object where each item is of form batch_size, (x, genre_embedding)
    # x should be a torch tensor of shape (batch_size, seq_len) where each element is an integer/word index
    # genre_embedding should be a torch tensor of shape (batch_size, genre_embed_size) where each embedding is the learned vector for that song

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        pass


def test(model, val_loader):
    model.eval()
    # test the model on validation set and return accuracy/F1 score or something
    # TODO


def inference(model, genre):
    model.eval()
    # genre is a torch tensor of shape (1, genre_embed_size) where each embedding is the learned vector for that song

    # return a torch tensor of shape (1, seq_len) where each element is an integer/word index
    # this is the generated song
    # TODO
