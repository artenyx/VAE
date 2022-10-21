import torch
import torch.nn as nn
from collections import OrderedDict



class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.net(x)  # See details in Kaiming He's original paper of ResNets


class Quantize(nn.Module):

    def __init__(self, size, code_dim):
        super().__init__()
        self.embedding = nn.Embedding(size, code_dim)
        # Initialization of embedding vectors: sampled from uniform distribution on [-1/K, 1/K], where K is the number of embedding vectors/categories
        self.embedding.weight.data.uniform_(-1./size, 1./size)
        self.code_dim = code_dim
        self.size = size

    # Define the mapping from z_e to z_e, with specific embedding vectors
    def forward(self, z):
        b, c, h, w = z.shape
        weight = self.embedding.weight

        flat_inputs = z.permute(0, 2, 3, 1).contiguous().view(-1, self.code_dim)
        # Compute the distances of each representation vector  and each embedding vector
        # by ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * torch.mm(flat_inputs, weight.t()) \
                    + (weight.t() ** 2).sum(dim=0, keepdim=True)
        # Find the nearest embedding vector for each representation vector
        encoding_indices = torch.max(-distances, dim=1)[1]
        encoding_indices = encoding_indices.view(b, h, w)
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2).contiguous()

        return quantized, (quantized - z).detach() + z, encoding_indices


class VectorQuantizedVAE(nn.Module):
    def __init__(self, code_dim, code_size):
        super().__init__()
        self.code_size = code_size

        # Encoder structure
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # Apply the Batch-normalization for 4D input: number of data point * number of features * H * W
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            ResidualBlock(256),  # Use the residual block as we defined above
            ResidualBlock(256),
        )
        # Initialize all embedding vectors by Quantize() and store it as codebook
        # Moreover the mapping from z_e to z_q is stored in the same codebook as well
        self.codebook = Quantize(code_size, code_dim)

        # Decoder structure
        self.decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    # Encode an image
    def encode_code(self, x):
        with torch.no_grad():
            x = 2 * x - 1  # From [-1,1] to [0,1]
            z = self.encoder(x)
            indices = self.codebook(z)[2]
            return indices

    # Decode an image
    def decode_code(self, latents):
        with torch.no_grad():
            latents = self.codebook.embedding(latents).permute(0, 3, 1, 2).contiguous()
            return self.decoder(latents).permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5

    def forward(self, x):
        z = self.encoder(x)
        e, e_st, _ = self.codebook(z)  # e_st: z_q
        x_tilde = self.decoder(e_st)  # Output image

        diff1 = torch.mean((z - e.detach()) ** 2)  # Commitment loss
        diff2 = torch.mean((e - z.detach()) ** 2)  # Loss from vector quantisation
        return x_tilde, diff1 + diff2

    def loss(self, x):
        x = 2 * x - 1
        x_tilde, diff = self(x)  # Initialize an object of this class, for which forward() is called by default
        recon_loss = nn.MSELoss(x_tilde, x)
        loss = recon_loss + diff
        return OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=diff)