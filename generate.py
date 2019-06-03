import os
import sys
import pickle
import argparse
import tempfile
import subprocess

import torch
import models
import data_utils
import train_helper

import numpy as np

from beam_search import beam_search, get_gen_fn
from config import BOS_IDX, EOS_IDX
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--save_file', '-s', type=str)
parser.add_argument('--vocab_file', '-v', type=str)
parser.add_argument('--input_file', '-i', type=str)
parser.add_argument('--ref_file', '-r', type=str)
parser.add_argument('--beam_size', '-bs', type=int, default=10)
args = parser.parse_args()


save_dict = torch.load(
    args.save_file,
    map_location=lambda storage,
    loc: storage)

config = save_dict['config']
checkpoint = save_dict['state_dict']
config.debug = True

with open(args.vocab_file, "rb") as fp:
    W, vocab = pickle.load(fp)
inv_vocab = {i: w for w, i in vocab.items()}

if config.decoder_type == "lstm":
    config.decoder_type = "lstm_z2y"
    config.ncode = None
    config.nclass = None

with train_helper.experiment(config, config.save_prefix) as e:
    e.log.info("vocab loaded from: {}".format(args.vocab_file))
    model = models.vgvae(
        vocab_size=len(vocab),
        embed_dim=e.config.edim if W is None else W.shape[1],
        embed_init=W,
        experiment=e)
    model.load(checkpointed_state_dict=checkpoint)
    e.log.info(model)

    semantics_input = []
    syntax_input = []
    e.log.info("loading from: {}".format(args.input_file))
    with open(args.input_file) as fp:
        for line in fp:
            seman_in, syn_in = line.strip().split("\t")
            semantics_input.append(
                [vocab.get(w.lower(), 0) for w in
                 seman_in.strip().split(" ")])
            syntax_input.append([vocab.get(w.lower(), 0) for w in
                                 syn_in.strip().split(" ")])
    e.log.info("#evaluation data: {}, {}".format(
        len(semantics_input),
        len(syntax_input)))

    tf = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    e.log.info('generation saving to {}'.format(tf.name))
    e.log.info('beam size: {}'.format(args.beam_size))
    for s1, _, m1, s2, _, m2, _, _, _, _, _ in \
        tqdm(data_utils.minibatcher(
                data1=np.array(semantics_input),
                tag1=np.array(semantics_input),
                data2=np.array(syntax_input),
                tag2=np.array(syntax_input),
                tag_bucket=None,
                batch_size=1,
                p_replace=0.,
                shuffle=False,
                p_scramble=0.)):
            with torch.no_grad():
                model.eval()
                semantics, semantics_mask, synatx, syntax_mask = \
                    model.to_tensors(s1, m1, s2, m2)
                _, yvecs = model.yencode(semantics, semantics_mask)
                _, zvecs = model.zencode(synatx, syntax_mask)

                ymean = model.mean1(yvecs)
                ymean = ymean / ymean.norm(dim=-1, keepdim=True)
                zmean = model.mean2(zvecs)

                generate_function = get_gen_fn(model.decode.step, ymean, zmean)
                initial_state = None
                if e.config.decoder_type.lower() == "yz_lstm":
                    init_vecs = model.decode.latent2init(
                        torch.cat([ymean, zmean], -1))
                    initial_state = tuple([h.unsqueeze(0).contiguous() for h in
                                           torch.chunk(init_vecs, 2, -1)])
                gen = beam_search(
                    initial_state=initial_state,
                    generate_function=generate_function,
                    start_id=BOS_IDX,
                    end_id=EOS_IDX,
                    beam_width=args.beam_size)[0][0]

                curr_gen = []
                for i in gen[1:]:
                    if i == EOS_IDX:
                        break
                    curr_gen.append(inv_vocab[int(i)])

                tf.write(" ".join(curr_gen))
                tf.write("\n")
    tf.flush()
    if args.ref_file is not None:
        e.log.info('running eval.py using reference file {}'
                   .format(args.ref_file))
        subprocess.run(
            ['python', 'eval.py', '-i', tf.name, '-r', args.ref_file],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=sys.stderr,
            stderr=sys.stdout)
