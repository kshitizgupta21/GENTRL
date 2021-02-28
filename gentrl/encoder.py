import torch
from torch import nn
from gentrl.tokenizer import encode, get_vocab_size


class RNNEncoder(nn.Module):
    def __init__(self,  hidden_size=256, num_layers=2, latent_size=50,
                 bidirectional=False):
        super(RNNEncoder, self).__init__()
        # here vocab size = 28
        # hidden_size = 256
        self.embs = nn.Embedding(num_embeddings=get_vocab_size(), embedding_dim=hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, 2 * latent_size))

    def encode(self, x, lens):
        """
        x: Tensor containing indices of tokenized SMILE strings in a batch
        lens: Tensor containing lengths of SMILE molecules in a batch
        Maps smiles onto a latent space
        """

        #tokens, lens = encode(sm_list)
        to_feed = tokens.transpose(1, 0)

        outputs, _ = self.rnn(self.embs(to_feed))
        outputs = outputs[lens, torch.arange(len(lens))]

        return self.final_mlp(outputs)
