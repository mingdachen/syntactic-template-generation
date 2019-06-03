# coding: utf-8

"""
Modified based on: https://github.com/ottokart/beam_search/blob/master/beam_search.py

"""

import numpy as np
import torch


class Node:
    def __init__(self, parent, state, value, cost):
        super(Node, self).__init__()
        self.value = value
        self.parent = parent # parent Node, None for root
        self.state = state if state is not None else None # recurrent layer hidden state
        self.cum_cost = parent.cum_cost + cost if parent else cost # e.g. -log(p) of sequence up to current node (including)
        self.length = 1 if parent is None else parent.length + 1
        # self.extras = extras # can hold, for example, attention weights
        self._sequence = None

    def to_sequence(self):
        # Return sequence of nodes from root to current node.
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence

    def to_sequence_of_values(self):
        return [s.value for s in self.to_sequence()]

    def to_sequence_of_extras(self):
        return [s.extras for s in self.to_sequence()]


def beam_search(initial_state, generate_function, start_id, end_id, beam_width=4, num_hypotheses=1, max_length=50):
    prev_history = [Node(parent=None, state=initial_state, value=start_id, cost=0.0)]
    hypotheses = []

    for _ in range(max_length):

        history = []
        for n in prev_history:
            if n.value == end_id or n.length == max_length:
                if n.length >= 3:
                    hypotheses.append(n)
            else:
                history.append(n)

        if not history or len(hypotheses) >= num_hypotheses:
            break

        state_t, p_t = list(zip(*[generate_function(n.state, n.value) for n in history]))
        Y_t = [np.argsort(p_t_n)[:beam_width] for p_t_n in p_t]  # no point in taking more than fits in the beam

        prev_history = []
        for Y_t_n, p_t_n, state_t_n, n in zip(Y_t, p_t, state_t, history):
            Y_nll_t_n = p_t_n[Y_t_n]

            for y_t_n, y_nll_t_n in zip(Y_t_n, Y_nll_t_n):
                n_new = Node(parent=n, state=state_t_n, value=y_t_n, cost=y_nll_t_n)
                prev_history.append(n_new)

        prev_history = sorted(prev_history, key=lambda n: n.cum_cost)[:beam_width] # may move this into loop to save memory

    hypotheses.sort(key=lambda n: n.cum_cost)
    result = [[hypo.to_sequence_of_values(), hypo.cum_cost] for hypo in hypotheses]
    return [res for res in result][:num_hypotheses]


def get_gen_fn(step_func, yvecs, zvecs):
    def generate_function(last_hidden_state, last_word):
        with torch.no_grad():
            last_word = torch.LongTensor(
                len(yvecs), 1).fill_(last_word).to(yvecs.device)
            next_state, next_word_prob, _, _ = \
                step_func(yvecs, zvecs, last_hidden_state, last_word)
        next_word_prob = next_word_prob.cpu().numpy()
        next_word_prob[next_word_prob < 0] = 0
        return next_state, - np.log(next_word_prob[0][0] + 1e-6)
    return generate_function
