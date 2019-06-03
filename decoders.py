import torch
import model_utils

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class decoder_base(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_init, dropout, log):
        super(decoder_base, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if embed_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embed_init))
            log.info(
                "{} initialized with pretrained word embedding".format(
                    type(self)))

    def greedy_decode(self, yvecs, zvecs, max_len):
        input_word = torch.ones(len(yvecs), 1).to(yvecs.device)
        batch_gen = []
        hidden_state = None
        for _ in range(max_len):
            hidden_state, _, input_word, _ = \
                self.step(yvecs, zvecs, hidden_state, input_word)
            batch_gen.append(input_word.detach().clone().cpu().numpy())
        batch_gen = np.concatenate(batch_gen, 1)
        return batch_gen


class lstm_z2y(decoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init,
                 ysize, zsize, mlp_hidden_size,
                 mlp_layer, hidden_size, dropout,
                 log, *args, **kwargs):
        super(lstm_z2y, self).__init__(
            vocab_size, embed_dim, embed_init, dropout, log)
        self.cell = nn.LSTM(
            zsize + embed_dim, hidden_size,
            bidirectional=False, batch_first=True)
        self.hid2vocab = nn.Linear(hidden_size + ysize, vocab_size)

    def forward(self, yvecs, zvecs, tgts, tgts_mask,
                *args, **kwargs):
        return self.teacher_force(yvecs, zvecs, tgts, tgts_mask)

    def pred(self, yvecs, zvecs, tgts, tgts_mask):
        bs, sl = tgts_mask.size()
        tgts_embed = self.dropout(self.embed(tgts.long()))
        ex_input_vecs = zvecs.unsqueeze(1).expand(-1, sl, -1)
        ex_output_vecs = yvecs.unsqueeze(1).expand(-1, sl, -1)

        input_vecs = torch.cat([tgts_embed, ex_input_vecs], -1)
        ori_output_seq, _ = model_utils.get_rnn_vecs(
            input_vecs, tgts_mask, self.cell, bidir=False, initial_state=None)
        output_seq = torch.cat([ori_output_seq, ex_output_vecs], -1)
        # batch size x seq len x vocab size
        pred = self.hid2vocab(self.dropout(output_seq))[:, :-1, :]
        return pred, input_vecs

    def teacher_force(self, yvecs, zvecs, tgts, tgts_mask):
        pred, pvecs = self.pred(yvecs, zvecs, tgts, tgts_mask)
        batch_size, seq_len, vocab_size = pred.size()

        pred = pred.contiguous().view(batch_size * seq_len, vocab_size)
        logloss = F.cross_entropy(
            pred, tgts[:, 1:].contiguous().view(-1).long(), reduction="none")

        logloss = (logloss.view(batch_size, seq_len) *
                   tgts_mask[:, 1:]).sum(-1) / tgts_mask[:, 1:].sum(-1)
        return logloss.mean(), pvecs

    def step(self, yvecs, zvecs, last_hidden_state, last_output):
        input_embed = self.embed(last_output.long())
        input_vec = torch.cat([input_embed, zvecs.unsqueeze(1)], -1)
        curr_output, hidden_state = self.cell(input_vec, last_hidden_state)
        curr_output = torch.cat([curr_output, yvecs.unsqueeze(1)], -1)
        curr_prob = F.softmax(self.hid2vocab(curr_output), -1)
        curr_prob[:, :, 0] = -10
        output_prob, input_word = curr_prob.max(-1)
        return hidden_state, curr_prob, input_word, output_prob


class lstm_y2z(decoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init,
                 ysize, zsize, mlp_hidden_size,
                 mlp_layer, hidden_size, dropout,
                 log, *args, **kwargs):
        super(lstm_y2z, self).__init__(
            vocab_size, embed_dim, embed_init, dropout, log)
        self.cell = nn.LSTM(
            ysize + embed_dim, hidden_size,
            bidirectional=False, batch_first=True)
        self.hid2vocab = nn.Linear(hidden_size + zsize, vocab_size)

    def forward(self, yvecs, zvecs, tgts, tgts_mask,
                *args, **kwargs):
        return self.teacher_force(yvecs, zvecs, tgts, tgts_mask)

    def pred(self, yvecs, zvecs, tgts, tgts_mask):
        bs, sl = tgts_mask.size()
        tgts_embed = self.dropout(self.embed(tgts.long()))
        ex_input_vecs = yvecs.unsqueeze(1).expand(-1, sl, -1)
        ex_output_vecs = zvecs.unsqueeze(1).expand(-1, sl, -1)

        input_vecs = torch.cat([tgts_embed, ex_input_vecs], -1)
        ori_output_seq, _ = model_utils.get_rnn_vecs(
            input_vecs, tgts_mask, self.cell, bidir=False, initial_state=None)
        output_seq = torch.cat([ori_output_seq, ex_output_vecs], -1)
        # batch size x seq len x vocab size
        pred = self.hid2vocab(self.dropout(output_seq))[:, :-1, :]
        return pred, torch.cat([tgts_embed, ex_output_vecs], -1)

    def teacher_force(self, yvecs, zvecs, tgts, tgts_mask):
        pred, pvecs = self.pred(yvecs, zvecs, tgts, tgts_mask)
        batch_size, seq_len, vocab_size = pred.size()

        pred = pred.contiguous().view(batch_size * seq_len, vocab_size)
        logloss = F.cross_entropy(
            pred, tgts[:, 1:].contiguous().view(-1).long(), reduction="none")

        logloss = (logloss.view(batch_size, seq_len) *
                   tgts_mask[:, 1:]).sum(-1) / tgts_mask[:, 1:].sum(-1)
        return logloss.mean(), pvecs

    def step(self, yvecs, zvecs, last_hidden_state, last_output):
        input_embed = self.embed(last_output.long())
        input_vec = torch.cat([input_embed, yvecs.unsqueeze(1)], -1)
        curr_output, hidden_state = self.cell(input_vec, last_hidden_state)
        curr_output = torch.cat([curr_output, zvecs.unsqueeze(1)], -1)
        curr_prob = F.softmax(self.hid2vocab(curr_output), -1)
        curr_prob[:, :, 0] = -10
        output_prob, input_word = curr_prob.max(-1)
        return hidden_state, curr_prob, input_word, output_prob


class lstm_yz(decoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init,
                 ysize, zsize, mlp_hidden_size,
                 mlp_layer, hidden_size, dropout,
                 log, *args, **kwargs):
        super(lstm_yz, self).__init__(
            vocab_size, embed_dim, embed_init, dropout, log)
        self.cell = nn.LSTM(
            zsize + ysize + embed_dim, hidden_size,
            bidirectional=False, batch_first=True)
        self.hid2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, yvecs, zvecs, tgts, tgts_mask,
                *args, **kwargs):
        return self.teacher_force(yvecs, zvecs, tgts, tgts_mask)

    def pred(self, yvecs, zvecs, tgts, tgts_mask):
        bs, sl = tgts_mask.size()
        tgts_embed = self.dropout(self.embed(tgts.long()))
        ex_input_vecs = zvecs.unsqueeze(1).expand(-1, sl, -1)
        ex_input2_vecs = yvecs.unsqueeze(1).expand(-1, sl, -1)

        input_vecs = torch.cat([tgts_embed, ex_input_vecs, ex_input2_vecs], -1)
        ori_output_seq, _ = model_utils.get_rnn_vecs(
            input_vecs, tgts_mask, self.cell, bidir=False, initial_state=None)
        output_seq = ori_output_seq
        # batch size x seq len x vocab size
        pred = self.hid2vocab(self.dropout(output_seq))[:, :-1, :]
        return pred, torch.cat([tgts_embed, ex_input_vecs], -1)

    def teacher_force(self, yvecs, zvecs, tgts, tgts_mask):
        pred, pvecs = self.pred(yvecs, zvecs, tgts, tgts_mask)
        batch_size, seq_len, vocab_size = pred.size()

        pred = pred.contiguous().view(batch_size * seq_len, vocab_size)
        logloss = F.cross_entropy(
            pred, tgts[:, 1:].contiguous().view(-1).long(), reduction="none")

        logloss = (logloss.view(batch_size, seq_len) *
                   tgts_mask[:, 1:]).sum(-1) / tgts_mask[:, 1:].sum(-1)
        return logloss.mean(), pvecs

    def step(self, yvecs, zvecs, last_hidden_state, last_output):
        input_embed = self.embed(last_output.long())
        input_vec = torch.cat(
            [input_embed, zvecs.unsqueeze(1), yvecs.unsqueeze(1)], -1)
        curr_output, hidden_state = self.cell(input_vec, last_hidden_state)
        curr_prob = F.softmax(self.hid2vocab(curr_output), -1)
        curr_prob[:, :, 0] = -10
        output_prob, input_word = curr_prob.max(-1)
        return hidden_state, curr_prob, input_word, output_prob


class yz_lstm(decoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init,
                 ysize, zsize, mlp_hidden_size,
                 mlp_layer, hidden_size, dropout,
                 log, *args, **kwargs):
        super(yz_lstm, self).__init__(
            vocab_size, embed_dim, embed_init, dropout, log)
        self.cell = nn.LSTM(
            embed_dim, hidden_size,
            bidirectional=False, batch_first=True)
        self.latent2init = nn.Linear(ysize + zsize, hidden_size * 2)
        self.hid2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, yvecs, zvecs, tgts, tgts_mask,
                *args, **kwargs):
        return self.teacher_force(yvecs, zvecs, tgts, tgts_mask)

    def pred(self, yvecs, zvecs, tgts, tgts_mask):
        bs, sl = tgts_mask.size()
        tgts_embed = self.dropout(self.embed(tgts.long()))
        init_vecs = self.latent2init(torch.cat([yvecs, zvecs], -1))

        if isinstance(self.cell, nn.LSTM):
            init_vecs = tuple([h.unsqueeze(0).contiguous() for h in
                              torch.chunk(init_vecs, 2, -1)])

        input_vecs = tgts_embed
        ori_output_seq, _ = model_utils.get_rnn_vecs(
            input_vecs, tgts_mask, self.cell, bidir=False, initial_state=init_vecs)
        output_seq = ori_output_seq
        # batch size x seq len x vocab size
        pred = self.hid2vocab(self.dropout(output_seq))[:, :-1, :]
        return pred, torch.cat(
            [tgts_embed, zvecs.unsqueeze(1).expand(-1, sl, -1)], -1)

    def teacher_force(self, yvecs, zvecs, tgts, tgts_mask):
        pred, pvecs = self.pred(yvecs, zvecs, tgts, tgts_mask)
        batch_size, seq_len, vocab_size = pred.size()

        pred = pred.contiguous().view(batch_size * seq_len, vocab_size)
        logloss = F.cross_entropy(
            pred, tgts[:, 1:].contiguous().view(-1).long(), reduction="none")

        logloss = (logloss.view(batch_size, seq_len) *
                   tgts_mask[:, 1:]).sum(-1) / tgts_mask[:, 1:].sum(-1)
        return logloss.mean(), pvecs

    def step(self, yvecs, zvecs, last_hidden_state, last_output):
        input_embed = self.embed(last_output.long())
        input_vec = input_embed
        curr_output, hidden_state = self.cell(input_vec, last_hidden_state)
        curr_prob = F.softmax(self.hid2vocab(curr_output), -1)
        curr_prob[:, :, 0] = -10
        output_prob, input_word = curr_prob.max(-1)
        return hidden_state, curr_prob, input_word, output_prob

    def greedy_decode(self, yvecs, zvecs, max_len):
        input_word = torch.ones(len(yvecs), 1).to(yvecs.device)
        batch_gen = []
        init_vecs = self.latent2init(torch.cat([yvecs, zvecs], -1))

        if isinstance(self.cell, nn.LSTM):
            init_vecs = tuple([h.unsqueeze(0).contiguous() for h in
                              torch.chunk(init_vecs, 2, -1)])
        hidden_state = init_vecs
        for _ in range(max_len):
            hidden_state, _, input_word, _ = \
                self.step(None, None, hidden_state, input_word)
            batch_gen.append(input_word.detach().clone().cpu().numpy())
        batch_gen = np.concatenate(batch_gen, 1)
        return batch_gen
