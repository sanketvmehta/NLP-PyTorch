import torch.nn as nn
import torch.nn.functional as F

class NGramLangModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_layer_dim = 128):
        super(NGramLangModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_layer_dim)
        self.linear2 = nn.Linear(hidden_layer_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim = 1)

        return log_probs
