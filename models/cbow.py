import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input):
        context_embeddings = self.embedding(input)
        bow_embeddings = torch.sum(context_embeddings, dim = 0).view(1, -1)
        out = self.linear(bow_embeddings)
        log_probs = F.log_softmax(out, dim = 1)

        return log_probs