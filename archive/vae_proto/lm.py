from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Parameter, LSTMCell, Embedding, LSTM, Linear


class LanguageModel(Module):

    def __init__(self, vocab_size, input_dim, hidden_dim, agenda_dim, num_layers=1,
                 drop_rate=0.6, tie=False, logger=None):
        super(LanguageModel, self).__init__()

        self.embed = Embedding(vocab_size, input_dim)
        self.decoder_rnn = LSTM(input_dim + agenda_dim, hidden_dim, num_layers=num_layers, dropout=drop_rate)
        self.decoder_out = Linear(hidden_dim, vocab_size)

        self.agenda_dim = agenda_dim
        self.logger = logger

        if tie:
            if hidden_dim != input_dim:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.embed.weight = self.encoder.weight

        # TODO init word embedding from pretrain
        self.init_weights(input_dim, hidden_dim, agenda_dim)

    def init_weights(self, input_dim, hidden_dim, agenda_dim):
        # kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'
        torch.nn.init.xavier_uniform(self.decoder_rnn.weight_ih_l0.data, gain=nn.init.calculate_gain('sigmoid'))
        torch.nn.init.orthogonal(self.decoder_rnn.weight_hh_l0.data, gain=nn.init.calculate_gain('sigmoid'))
        self.decoder_rnn.bias.data.fill_(0)

        # embedding uniform
        torch.nn.init.xavier_uniform(self.embed.weight.data, gain=nn.init.calculate_gain('linear'))

        # Linear kernel_initializer='glorot_uniform'
        torch.nn.init.xavier_uniform(self.decoder_out.weight.data, gain=nn.init.calculate_gain('linear'))

    def _encoder_output(self, batch_size):
        return tile_state(self.agenda, batch_size)

    def per_instance_losses(self, examples):
        batch_size = len(examples)
        decoder_input = TrainDecoderInput(examples, self.vocab)
        encoder_output = self._encoder_output(batch_size)
        return self.train_decoder.per_instance_losses(encoder_output, decoder_input)

    def loss(self, examples, train_step):
        """Compute training loss.

        Args:
            examples (list[list[unicode]])

        Returns:
            Variable: a scalar
        """
        batch_size = len(examples)
        decoder_input = TrainDecoderInput(examples, self.vocab)
        encoder_output = self._encoder_output(batch_size)
        return self.train_decoder.loss(encoder_output, decoder_input)

    def generate(self, num_samples, decode_method='argmax'):
        examples = range(num_samples)
        prefix_hints = [[]] * num_samples  # none
        encoder_output = self._encoder_output(num_samples)
        if decode_method == 'sample':
            output_beams, decoder_traces = self.sample_decoder.decode(examples, encoder_output,
                                                                      beam_size=1, prefix_hints=prefix_hints)
        elif decode_method == 'argmax':
            value_estimators = []
            beam_size = 1
            sibling_penalty = 0.
            output_beams, decoder_traces = self.beam_decoder.decode(examples, encoder_output,
                                                                    weighted_value_estimators=value_estimators,
                                                                    beam_size=beam_size, prefix_hints=prefix_hints,
                                                                    sibling_penalty=sibling_penalty)
        else:
            raise ValueError(decode_method)

        return [beam[0] for beam in output_beams]


class Encoder(Module):

    def __init__(self, token_embedder, hidden_dim, agenda_dim, num_layers, rnn_cell_factory):
        super(Encoder, self).__init__()

        self.token_embedder = token_embedder
        self.word_vocab = token_embedder.vocab
        self.hidden_dim = hidden_dim
        self.agenda_dim = agenda_dim
        self.num_layers = num_layers

        self.source_encoder = MultiLayerSourceEncoder(
            token_embedder.embed_dim, hidden_dim, num_layers, rnn_cell_factory)

    def preprocess(self, examples):
        return SequenceBatch.from_sequences(examples, self.word_vocab)

    def forward(self, examples_seq_batch):
        embeds = self.token_embedder.embed_seq_batch(examples_seq_batch)
        source_encoder_output = self.source_encoder(embeds.split())
        return source_encoder_output

    def make_agenda(self, encoder_output):
        agenda = torch.cat(encoder_output.final_states, 1)
        return agenda


class EncoderNoiser(Module):

    def __init__(self, encoder, kl_weight_steps, kl_weight_rate, kl_weight_cap):
        super(EncoderNoiser, self).__init__()

        self.encoder = encoder
        self.noise_mu = 0
        self.noise_sigma = 1
        self.kl_weight_steps = kl_weight_steps
        self.kl_weight_rate = kl_weight_rate
        self.kl_weight_cap = kl_weight_cap

    def preprocess(self, examples):
        return self.encoder.preprocess(examples)

    def kl_penalty(self, agenda):
        """
        Computes KL penalty given encoder output
        """
        batch_size, agenda_dim = agenda.size()
        return 0.5 * torch.sum(torch.pow(agenda, 2)) * self.noise_sigma / batch_size

    def kl_weight(self, curr_step):
        """
        Compute KL penalty weight
        """

        sigmoid = lambda x, k: 1 / (1 + np.e ** (-k * (2 * x - 1)))
        x = curr_step / float(self.kl_weight_steps)
        return self.kl_weight_cap * sigmoid(x, self.kl_weight_rate)

    def forward(self, examples_seq_batch):
        source_encoder_output = self.encoder(examples_seq_batch)
        agenda = self.encoder.make_agenda(source_encoder_output)
        means = self.noise_mu * torch.ones(agenda.size())
        std = self.noise_sigma * torch.ones(agenda.size())
        noise = GPUVariable(torch.normal(
            means=means, std=std))  # unit Gaussian
        return agenda, agenda + noise


class NoisyLanguageModel(LanguageModel):

    def __init__(self, token_embedder, hidden_dim, agenda_dim, num_layers, kl_weight_steps, kl_weight_rate,
                 kl_weight_cap, dci_keep_rate, logger):
        super(NoisyLanguageModel, self).__init__(
            token_embedder, hidden_dim, agenda_dim, num_layers, logger)

        # encoder
        encoder = Encoder(token_embedder, hidden_dim, agenda_dim,
                          num_layers, rnn_cell_factory=LSTMCell)
        self.encoder = EncoderNoiser(encoder, kl_weight_steps, kl_weight_rate, kl_weight_cap)

        # decoder dropout
        self.dci_keep_rate = dci_keep_rate

    def loss(self, examples, train_step):
        """Compute training loss.

        Args:
            examples (list[list[unicode]])

        Returns:
            Variable: a scalar
        """

        enc_input = self.encoder.preprocess(examples)
        agenda, noised_agenda = self.encoder(enc_input)
        kl = self.encoder.kl_penalty(agenda)
        kl_wt = self.encoder.kl_weight(train_step)

        # kl and kl_wt are accessible only here, log them
        # note that logger will only _really_ log if sl(train_steps) is true
        kl_ = kl.data.cpu().numpy()[0]
        self.logger('kl_penalty', kl_, train_step)
        self.logger('kl_weight', kl_wt, train_step)

        decoder_input = DropoutTrainDecoderInput(examples, self.vocab, self.dci_keep_rate)
        return kl_wt * kl + self.train_decoder.loss(noised_agenda, decoder_input)

    def _interpolate_vectors(self, v_a, v_b, steps=5):
        lambdas = np.linspace(0, 1, steps)
        interps = [(1 - l) * v_a + l * v_b for l in lambdas]
        return interps

    def _interpolate_examples(self, ex_a, ex_b):
        """
        Args:
            [unicode], [unicode]

        Returns:
            [[unicode]]
        """

        examples = [ex_a, ex_b]
        enc_input = self.encoder.preprocess(examples)
        agenda, _ = self.encoder(enc_input)
        agenda_ = agenda.data.cpu().numpy()
        agendas = self._interpolate_vectors(agenda_[0, :], agenda_[1, :])
        samples = []
        for i, ag_ in enumerate(agendas):
            ag = GPUVariable(torch.FloatTensor(ag_.reshape(1, self.agenda_dim)))
            # beam, _ = self.sample_decoder.decode(
            #    [0], ag, beam_size=1, prefix_hints=[[]])
            beam, _ = self.beam_decoder.decode(
                [0], ag, weighted_value_estimators=[], beam_size=1, prefix_hints=[[]], sibling_penalty=0)
            samples.append(beam[0][0])
        return samples
