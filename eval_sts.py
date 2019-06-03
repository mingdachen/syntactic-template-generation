import nltk
import pickle
import argparse

import torch
import models
import data_utils
import train_helper

import numpy as np

from tqdm import tqdm
from scipy.stats import pearsonr


MAX_LEN = 30
batch_size = 1000


parser = argparse.ArgumentParser()
parser.add_argument('--save_file', '-s', type=str)
parser.add_argument('--vocab_file', '-v', type=str)
parser.add_argument('--sts_file', '-d', type=str, default="sts-test.csv")
args = parser.parse_args()


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

sent1 = []
sent2 = []
gold_score = []
with open(args.sts_file) as fp:
    for i, line in enumerate(fp):
        d = line.strip().split("\t")
        score, s1, s2 = d[4], d[5], d[6]
        sent1.append(nltk.word_tokenize(s1.lower()))
        sent2.append(nltk.word_tokenize(s2.lower()))
        gold_score.append(float(score))

with train_helper.experiment(config, config.save_prefix) as e:
    e.log.info("vocab loaded from: {}".format(args.vocab_file))
    e.log.info("data loaded from: {}".format(args.sts_file))
    model = models.vgvae(
        vocab_size=len(vocab),
        embed_dim=e.config.edim if W is None else W.shape[1],
        embed_init=W,
        experiment=e)
    model.eval()
    model.load(checkpointed_state_dict=checkpoint)
    e.log.info(model)

    def encode(d):
        global vocab
        new_d = [[vocab.get(w, 0) for w in s] for s in d]
        all_y_vecs = []
        all_z_vecs = []

        for s1, _, m1, s2, _, m2, _, _, _, _, _ in \
            tqdm(data_utils.minibatcher(
                    data1=np.array(new_d),
                    tag1=np.array(new_d),
                    data2=np.array(new_d),
                    tag2=np.array(new_d),
                    tag_bucket=None,
                    batch_size=100,
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

    s1y, s1z = encode(sent1)
    s2y, s2z = encode(sent2)
    yscore = pearsonr(cosine_similarity(s1y, s2y), gold_score)[0]
    zscore = pearsonr(cosine_similarity(s1z, s2z), gold_score)[0]
    e.log.info("y score: {:.4f}, z score: {:.4f}".format(yscore, zscore))
