import pickle
import argparse
import collections

import tree
import torch
import models
import data_utils
import train_helper

import numpy as np

from tqdm import tqdm


MAX_LEN = 30
batch_size = 1000


parser = argparse.ArgumentParser()
parser.add_argument('--save_file', '-s', type=str)
parser.add_argument('--vocab_file', '-v', type=str)
parser.add_argument('--data_dir', '-d', type=str)
args = parser.parse_args()


def _brackets_helper(node, i, result):
    i0 = i
    if len(node.children) > 0:
        for child in node.children:
            i = _brackets_helper(child, i, result)
        j0 = i
        if len(node.children[0].children) > 0: # don't count preterminals
            result[node.label, i0, j0] += 1
    else:
        j0 = i0 + 1
    return j0


def brackets(t):
    result = collections.defaultdict(int)
    _brackets_helper(t.root, 0, result)
    return result


def cosine_similarity(v1, v2):
    prod = (v1 * v2).sum(-1)
    v1_norm = (v1 ** 2).sum(-1) ** 0.5
    v2_norm = (v2 ** 2).sum(-1) ** 0.5
    return prod / (v1_norm * v2_norm)


save_dict = torch.load(
    args.save_file,
    map_location=lambda storage,
    loc: storage)

config = save_dict['config']
checkpoint = save_dict['state_dict']
config.debug = True

with open(args.vocab_file, "rb") as fp:
    W, vocab = pickle.load(fp)

with train_helper.experiment(config, config.save_prefix) as e:
    e.log.info("vocab loaded from: {}".format(args.vocab_file))
    model = models.vgvae(
        vocab_size=len(vocab),
        embed_dim=e.config.edim if W is None else W.shape[1],
        embed_init=W,
        experiment=e)
    model.eval()
    model.load(checkpointed_state_dict=checkpoint)
    e.log.info(model)

    def encode(d):
        global vocab, batch_size
        new_d = [[vocab.get(w, 0) for w in s.split(" ")] for s in d]
        all_y_vecs = []
        all_z_vecs = []

        for s1, _, m1, s2, _, m2, _, _, _, _, _ in \
            tqdm(data_utils.minibatcher(
                    data1=np.array(new_d),
                    tag1=np.array(new_d),
                    data2=np.array(new_d),
                    tag2=np.array(new_d),
                    tag_bucket=None,
                    batch_size=batch_size,
                    p_replace=0.,
                    shuffle=False,
                    p_scramble=0.)):
                with torch.no_grad():
                    semantics, semantics_mask, synatx, syntax_mask = \
                        model.to_tensors(s1, m1, s2, m2)
                    _, yvecs = model.yencode(semantics, semantics_mask)
                    _, zvecs = model.zencode(synatx, syntax_mask)

                    ymean = model.mean1(yvecs)
                    ymean = ymean / ymean.norm(dim=-1, keepdim=True)
                    zmean = model.mean2(zvecs)

                    all_y_vecs.append(ymean.cpu().numpy())
                    all_z_vecs.append(zmean.cpu().numpy())
        return np.concatenate(all_y_vecs), np.concatenate(all_z_vecs)

    y_tot_pred = {i: [] for i in range(1, MAX_LEN)}
    tot_label = {i: [] for i in range(1, MAX_LEN)}
    y_results_len = {i: {"match_count": 0,
                         "parse_count": 0,
                         "gold_count": 0,
                         "best_f1": []}
                     for i in range(1, MAX_LEN)}

    z_tot_pred = {i: [] for i in range(1, MAX_LEN)}
    z_results_len = {i: {"match_count": 0,
                         "parse_count": 0,
                         "gold_count": 0,
                         "best_f1": []}
                     for i in range(1, MAX_LEN)}

    tag2num = {}
    for i in range(1, MAX_LEN):
        e.log.info("*" * 25 + " Length: {} ".format(i) + "*" * 25)
        cand_sents = []
        test_sents = []
        cand_pos = []
        cand_parse = []
        test_pos = []
        test_parse = []
        with open(args.data_dir + "/{}_candidates.txt".format(i)) as cf, \
                open(args.data_dir + "/{}_test.txt".format(i)) as tf:
            for line in cf:
                sent, pos, parse = line.strip().split("\t")
                cand_pos.append(pos.strip())
                cand_sents.append(sent.strip())
                cand_parse.append(parse.strip())
            for line in tf:
                sent, pos, parse = line.strip().split("\t")
                test_pos.append(pos.strip())
                test_sents.append(sent.strip())
                test_parse.append(parse.strip())
        y_test_vecs, z_test_vecs = encode(test_sents)
        y_cand_vecs, z_cand_vecs = encode(cand_sents)
        e.log.info("#query: {}, #candidate: {}"
                   .format(len(test_pos), len(cand_pos)))
        pbar = tqdm(zip(test_pos, test_parse, y_test_vecs, z_test_vecs))
        for curr_label, gold_parse, y_test_vec, z_test_vec in pbar:
            gold = tree.Tree.from_str(gold_parse)
            gold_brackets = brackets(gold)
            y_results_len[i]["gold_count"] += sum(gold_brackets.values())
            z_results_len[i]["gold_count"] += sum(gold_brackets.values())
            idx = cosine_similarity(
                y_test_vec[None, :], y_cand_vecs).argmax(-1)
            y_best_pred = cand_pos[idx]
            y_best_parse = cand_parse[idx]
            idx = cosine_similarity(
                z_test_vec[None, :], z_cand_vecs).argmax(-1)
            z_best_pred = cand_pos[idx]
            z_best_parse = cand_parse[idx]

            curr_label_ = []
            for t in curr_label.strip().split(" "):
                if t not in tag2num:
                    tag2num[t] = len(tag2num)
                curr_label_.append(tag2num[t])
            tot_label[i].extend(curr_label_)
            y_best_pred_ = []
            for t in y_best_pred.strip().split(" "):
                if t not in tag2num:
                    tag2num[t] = len(tag2num)
                y_best_pred_.append(tag2num[t])
            y_tot_pred[i].extend(y_best_pred_)
            z_best_pred_ = []
            for t in z_best_pred.strip().split(" "):
                if t not in tag2num:
                    tag2num[t] = len(tag2num)
                z_best_pred_.append(tag2num[t])
            z_tot_pred[i].extend(z_best_pred_)

            parse = tree.Tree.from_str(y_best_parse)
            parse_brackets = brackets(parse)
            delta_parse_count = sum(parse_brackets.values())

            delta_match_count = 0
            for bracket, count in parse_brackets.items():
                delta_match_count += min(count, gold_brackets[bracket])
            y_curr_f1 = (2. / (y_results_len[i]["gold_count"] /
                         float(y_results_len[i]["match_count"] + delta_match_count) +
                       (y_results_len[i]["parse_count"] + delta_parse_count) /
                       float(y_results_len[i]["match_count"] + delta_match_count))) \
                if float(y_results_len[i]["match_count"] + delta_match_count) else 0

            y_results_len[i]["match_count"] += delta_match_count
            y_results_len[i]["parse_count"] += delta_parse_count

            parse = tree.Tree.from_str(z_best_parse)
            parse_brackets = brackets(parse)
            delta_parse_count = sum(parse_brackets.values())

            delta_match_count = 0
            for bracket, count in parse_brackets.items():
                delta_match_count += min(count, gold_brackets[bracket])
            z_curr_f1 = (2. / (z_results_len[i]["gold_count"] /
                         float(z_results_len[i]["match_count"] + delta_match_count) +
                       (z_results_len[i]["parse_count"] + delta_parse_count) /
                       float(z_results_len[i]["match_count"] + delta_match_count))) \
                if float(z_results_len[i]["match_count"] + delta_match_count) else 0

            z_results_len[i]["match_count"] += delta_match_count
            z_results_len[i]["parse_count"] += delta_parse_count

            pbar.set_description(
                "y - curr acc: {:.4f}, f1: {:.4f}, "
                "z - curr acc: {:.4f}, f1: {:.4f}".format(
                    (np.array(y_tot_pred[i]) == np.array(tot_label[i])
                        .astype("float32")).mean(), y_curr_f1,
                    (np.array(z_tot_pred[i]) == np.array(tot_label[i])
                        .astype("float32")).mean(), z_curr_f1))
        pbar.close()

        e.log.info(
            "y - curr acc: {:.4f}, f1: {:.4f}, "
            "z - curr acc: {:.4f}, f1: {:.4f}".format(
                (np.array(y_tot_pred[i]) == np.array(tot_label[i])
                    .astype("float32")).mean(), y_curr_f1,
                (np.array(z_tot_pred[i]) == np.array(tot_label[i])
                    .astype("float32")).mean(), z_curr_f1))

    e.log.info("*" * 25 + " EVAL y " + "*" * 25)
    e.log.info("*" * 25 + " POS Acc " + "*" * 25)
    for i in range(1, MAX_LEN):
        e.log.info("length: {}, acc: {:.4f}"
                   .format(i, (np.array(y_tot_pred[i]) ==
                               np.array(tot_label[i]).astype("float32")).mean()))

    e.log.info("*" * 25 + " Labeled F1 " + "*" * 25)
    tot_match = tot_parse = tot_gold = 0
    for lens, d in sorted(y_results_len.items()):
        e.log.info("length: {}, F1: {:.4f}"
                   .format(lens, (2. / (d["gold_count"] / float(d["match_count"]) +
                                        d["parse_count"] / float(d["match_count"])))))
        tot_match += d["match_count"]
        tot_parse += d["parse_count"]
        tot_gold += d["gold_count"]

    all_pred = sum(y_tot_pred.values(), [])
    all_label = sum(tot_label.values(), [])
    e.log.info("POS Acc: {:.4f}".format(
        (np.array(all_pred) == np.array(all_label)).astype("float32").mean()))

    e.log.info(
        "Labeled F1: {:.4f}".format(
            2. / (tot_gold / float(tot_match) + tot_parse / float(tot_match))))

    e.log.info("*" * 25 + " EVAL z " + "*" * 25)
    e.log.info("*" * 25 + " POS Acc " + "*" * 25)

    for i in range(1, MAX_LEN):
        e.log.info("length: {}, acc: {:.4f}"
                   .format(i, (np.array(z_tot_pred[i]) ==
                               np.array(tot_label[i])
                               .astype("float32")).mean()))

    e.log.info("*" * 25 + " Labeled F1 " + "*" * 25)
    tot_match = tot_parse = tot_gold = 0
    for lens, d in sorted(z_results_len.items()):
        e.log.info("length: {}, F1: {:.4f}"
                   .format(lens, (2. / (d["gold_count"] / float(d["match_count"]) +
                                        d["parse_count"] / float(d["match_count"])))))
        tot_match += d["match_count"]
        tot_parse += d["parse_count"]
        tot_gold += d["gold_count"]

    all_pred = sum(z_tot_pred.values(), [])
    all_label = sum(tot_label.values(), [])
    e.log.info("POS Acc: {:.4f}".format(
        (np.array(all_pred) == np.array(all_label)).astype("float32").mean()))

    e.log.info(
        "Labeled F1: {:.4f}".format(2. / (tot_gold / float(tot_match) +
                                          tot_parse / float(tot_match))))
