import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

from models.ngramlm import NGramLangModel
from models.cbow import CBOW
from data.sample_data import *

torch.manual_seed(1)

model = 2

if model == 1:
    # NGramLangModel
    print("N-Gram Language Model!")
    context_size = 2
    embedding_dim = 10

    test_sentence = test_sentence.split()

    # Generate tuples with input as context word and label as target word
    trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
                for i in range(len(test_sentence) - 2)]

    vocab = set(test_sentence)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    losses = []
    loss_fn = nn.NLLLoss()
    model = NGramLangModel(vocab_size=len(vocab), context_size=context_size, embedding_dim=embedding_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.2)

    for epoch in range(10):
        total_loss = torch.Tensor([0])

        for context, target_word in trigrams:

            context_ids = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_ids))

            model.zero_grad()

            log_probs = model(context_var)

            loss = loss_fn(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target_word]])))

            loss.backward()
            optimizer.step()

            total_loss += loss.data
        losses.append(total_loss)

    print(losses)

elif model == 2:
    # CBoW
    print("Continuous Bag-of-Words!")
    context_size = 2
    embedding_dim = 10

    raw_text = raw_text.split()

    vocab = set(raw_text)
    vocab_size = len(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    data = []

    for i in range(2, len(raw_text) - 2):
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))
    print(data[:5])

    losses = []
    loss_fn = nn.NLLLoss()
    model = CBOW(vocab_size=vocab_size, embedding_dim=embedding_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.2)

    for epoch in range(10):
        total_loss = torch.Tensor([0])

        for context, target_word in data:
            context_ids = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_ids))

            model.zero_grad()

            log_probs = model(context_var)

            loss = loss_fn(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target_word]])))

            loss.backward()
            optimizer.step()

            total_loss += loss.data
        losses.append(total_loss)

    print(losses)

else:
    print("Option not found!")

