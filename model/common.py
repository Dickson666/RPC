import torch;
import torch.nn as nn;

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, latent) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fch = nn.Linear(hidden, hidden)
        self.fc2_mean = nn.Linear(hidden, latent)
        self.fc2_logvar = nn.Linear(hidden, latent)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.drop(x)
        x = self.relu(self.fch(x))
        # x = self.drop(x)
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden, latent) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent, hidden)
        self.fch = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.drop(x)
        x = self.relu(self.fch(x))
        # x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        return x