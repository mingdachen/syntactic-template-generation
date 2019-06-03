import torch
import model_utils
import torch.nn as nn
import torch.nn.functional as F


class encoder_base(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_init, dropout, log,
                 *args, **kwargs):
        super(encoder_base, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if embed_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embed_init))
            log.info(
                "{} initialized with pretrained word embedding".format(
                    type(self)))


class word_avg(encoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init, dropout, log,
                 *args, **kwargs):
        super(word_avg, self).__init__(
            vocab_size, embed_dim, embed_init, dropout, log)

    def forward(self, inputs, mask):
        input_vecs = self.dropout(self.embed(inputs.long()))
        sum_vecs = (input_vecs * mask.unsqueeze(-1)).sum(1)
        avg_vecs = sum_vecs / mask.sum(1, keepdim=True)
        return input_vecs, avg_vecs


class bilstm(encoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init, hidden_size,
                 dropout, log, *args, **kwargs):
        super(bilstm, self).__init__(
            vocab_size, embed_dim, embed_init, dropout, log)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, inputs, mask, temp=None):
        input_vecs = self.dropout(self.embed(inputs.long()))
        outputs, _ = model_utils.get_rnn_vecs(
            input_vecs, mask, self.lstm, bidir=True)
        outputs = self.dropout(outputs) * mask.unsqueeze(-1)
        sent_vec = outputs.sum(1) / mask.sum(1, keepdim=True)
        return input_vecs, sent_vec


class bilstm_lc(encoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init, hidden_size,
                 dropout, nclass, ncode, mlp_hidden_size, mlp_layer, log,
                 *args, **kwargs):
        super(bilstm_lc, self).__init__(
            vocab_size, embed_dim, embed_init, dropout, log)
        self.lstm = nn.LSTM(
            embed_dim // ncode * ncode, hidden_size,
            bidirectional=True, batch_first=True)
        self.lc = nn.ModuleList(
            [model_utils.get_mlp(
                embed_dim, hidden_size, nclass, mlp_layer, dropout)
                for _ in range(ncode)])
        self.lc_embed = nn.ModuleList(
            [nn.Embedding(nclass, embed_dim // ncode) for _ in range(ncode)])

    def forward(self, inputs, mask, temp=None):
        input_vecs = self.dropout(self.embed(inputs.long()))
        lc_vecs = []
        for proj, emb in zip(self.lc, self.lc_embed):
            prob = F.softmax(proj(input_vecs), -1)
            lc_vecs.append(torch.matmul(prob, emb.weight))
        input_vecs = self.dropout(torch.cat(lc_vecs, -1))
        outputs, _ = model_utils.get_rnn_vecs(
            input_vecs, mask, self.lstm, bidir=True)
        outputs = self.dropout(outputs) * mask.unsqueeze(-1)
        sent_vec = outputs.sum(1) / mask.sum(1, keepdim=True)
        return input_vecs, sent_vec
