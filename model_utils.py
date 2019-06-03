import torch
import torch.nn as nn


def get_mlp(input_size, hidden_size, output_size, n_layer, dropout):
    if n_layer == 0:
        proj = nn.Sequential(
            nn.Linear(input_size, output_size))
    else:
        proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout))

        for i in range(n_layer - 1):
            proj.add_module(
                str(len(proj)),
                nn.Linear(hidden_size, hidden_size))
            proj.add_module(str(len(proj)), nn.ReLU())
            proj.add_module(str(len(proj)), nn.Dropout(dropout))

        proj.add_module(
            str(len(proj)),
            nn.Linear(hidden_size, output_size))
    return proj


def get_rnn_vecs(
        inputs,
        mask,
        cell,
        bidir=False,
        initial_state=None,
        get_last=False,
        filter_zero_length=False):
    """
    Args:
    inputs: batch_size x seq_len x n_feat
    mask: batch_size x seq_len
    initial_state: batch_size x num_layers x hidden_size
    cell: GRU/LSTM/RNN
    """
    if bidir:
        seq_lengths = torch.sum(mask, dim=-1)
        sorted_len, sorted_idx = seq_lengths.sort(0, descending=True)
        sorted_inputs = inputs[sorted_idx.long()]
        sorted_len = sorted_len.long().cpu().numpy()
        if filter_zero_length:
            sorted_len[sorted_len == 0] = 1
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len, batch_first=True)
        if initial_state is not None:
            if isinstance(cell, torch.nn.LSTM):
                initial_state = \
                    (initial_state[0].index_select(1, sorted_idx.long()),
                     initial_state[1].index_select(1, sorted_idx.long()))
            else:
                initial_state = \
                    initial_state.index_select(1, sorted_idx.long())
        out, hid = cell(packed_seq, hx=initial_state)
        unpacked, unpacked_len = \
            torch.nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()]
        if get_last:
            if isinstance(hid, tuple):
                if bidir:
                    hid = tuple([torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
                                 for h in hid])
                hid = \
                    (hid[0].index_select(1, original_idx.long()),
                     hid[1].index_select(1, original_idx.long()))
            else:
                if bidir:
                    hid = torch.cat(
                        [hid[0:hid.size(0):2], hid[1:hid.size(0):2]], 2)
                hid = hid.index_select(1, original_idx.long())
    else:
        output_seq, hid = cell(inputs, hx=initial_state)
    return output_seq, hid


def gauss_kl_div(mean, logvar, eps=1e-8):
    """KL(p||N(0,1))
    args:
    mean: batch size x * x dimension
    logvar: batch size x * x dimension

    return:
    KL divergence: batch size x *
    """
    return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(-1)
