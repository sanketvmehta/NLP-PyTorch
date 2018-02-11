import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PosTagger(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_tags, dropout = 0,
                 bias = True, bidirectional = False):
        super(PosTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim,
                            num_layers = num_layers, bias = bias, dropout = dropout,
                            bidirectional = bidirectional)
        self.linear = nn.Linear(hidden_dim, num_tags)
        self.hidden = self.initialize_hidden()

    def initialize_hidden(self):

         return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                 autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, input):

        embeddings = self.embedding(input)
        lstm_input = embeddings.view(len(input), 1, -1)
        lstm_out, self.hidden = self.lstm(lstm_input, self.hidden)
        out = self.linear(lstm_out.view(len(input), -1))
        log_probs = F.log_softmax(out, dim = 1)

        return  log_probs