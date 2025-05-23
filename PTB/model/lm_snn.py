import torch
import torch.nn as nn
import math
from torch.autograd import Variable


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_idx, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse
                                      )
    return X


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class LMSNN(nn.Module):
    def __init__(self,
                 rnn_type,
                 nlayers,
                 emb_dim,
                 hidden_dim,
                 vocab_size,
                 dropout_words,
                 dropout_embedding,
                 dropout_forward,
                 dropout,
                 spiking_neuron=None,
                 args=None,
                 ):
        super(LMSNN, self).__init__()

        self.args = args

        # language model specifics
        self.nlayers = nlayers
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # dropout initializations
        self.dropout_words = dropout_words
        self.dropout_embedding = dropout_embedding
        self.dropout_forward = dropout_forward
        self.dropout = dropout

        # input and output layers
        self.locked_dropout = LockedDropout()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.decoder = nn.Linear(emb_dim, vocab_size)
        self.return_state = True

        self.init_weights(initrange=0.1)

        # Tie weights of embedding and decoder
        self.decoder.weight = self.embeddings.weight

        # RNN model definition
        self.rnn_type = rnn_type

        self.linears = [nn.Linear(emb_dim if l == 0 else hidden_dim,
                                  emb_dim if l == nlayers - 1 else hidden_dim,
                                  )
                        for l in range(nlayers)]
        self.snns = [spiking_neuron(neuron_num=emb_dim if l == nlayers - 1 else hidden_dim) for l in range(nlayers)]


        if self.rnn_type == 'TE-N':
            self.TE = nn.Parameter(torch.zeros(max(emb_dim,hidden_dim),self.snns[0].time_step))
            nn.init.normal_(self.TE, 0.01, 0.01)
            for i in range(nlayers):
                self.snns[i].TE = self.TE
        elif self.rnn_type == 'TE-R':
            dt = 0.1 / (1e2 ** (torch.arange(max(emb_dim,hidden_dim)).float()/ max(emb_dim,hidden_dim)))
            for i in range(nlayers):
                self.snns[i].fre = nn.Parameter((torch.rand(max(emb_dim,hidden_dim))+0.99) * math.pi * 2)
                self.snns[i].dt = dt


        self.linears = nn.ModuleList(self.linears)
        self.snns = nn.ModuleList(self.snns)


    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        self.decoder.bias.data.fill_(0)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return [(weight.new_zeros(batch_size,
                                  self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                 weight.new_zeros(batch_size,
                                  self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                 weight.new_full((batch_size,
                                  self.emb_dim if l == self.nlayers - 1 else self.hidden_dim), 0.5)  # v, y, thresh
                 )
                for l in range(self.nlayers)]

    def forward(self, inputs, state):
        return self.spk_forward(inputs, state)

    def spk_forward(self, inputs, state): # inputs: [T, B, N]
        embedded = embedded_dropout(self.embeddings, inputs,
                                    dropout=self.dropout_words if self.training else 0)

        embedded = self.locked_dropout(
            embedded, dropout=self.dropout_embedding)

        # rnn forward
        new_states = []
        hiddens = embedded
        self.loss = []
        for l, (linear, snn) in enumerate(zip(self.linears, self.snns)):
            hiddens = linear(hiddens)
            hiddens, final_states = snn(hiddens, state[l])
            new_states.append(final_states)
            if l != self.nlayers - 1:
                hiddens = self.locked_dropout(
                    hiddens, dropout=self.dropout_forward)
        # decoder forward
        hiddens = self.locked_dropout(hiddens, self.dropout)

        decoded = self.decoder(hiddens)
        return decoded, new_states


