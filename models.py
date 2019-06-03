import os
import torch
import model_utils
import encoders
import decoders

import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from von_mises_fisher import VonMisesFisher
from decorators import auto_init_args, auto_init_pytorch

MAX_LEN = 32


class base(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_init, experiment):
        super(base, self).__init__()
        self.expe = experiment
        self.eps = self.expe.config.eps
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def pos_loss(self, mask, vecs, func):
        batch_size, seq_len = mask.size()
        # batch size x seq len x MAX LEN
        logits = func(vecs)
        if MAX_LEN - seq_len:
            padded = torch.zeros(batch_size, MAX_LEN - seq_len).to(mask.device)
            new_mask = 1 - torch.cat([mask, padded], -1)
        else:
            new_mask = 1 - mask
        new_mask = new_mask.unsqueeze(1).expand_as(logits)
        logits.masked_fill_(new_mask.byte(), -float('inf'))
        loss = F.softmax(logits, -1)[:, np.arange(int(seq_len)),
                                     np.arange(int(seq_len))]
        loss = -(loss + self.eps).log() * mask

        loss = loss.sum(-1) / mask.sum(1)
        return loss.mean()

    def sample_gaussian(self, mean, logvar):
        sample = mean + torch.exp(0.5 * logvar) * \
            logvar.new_empty(logvar.size()).normal_()
        return sample

    def to_tensor(self, inputs):
        if torch.is_tensor(inputs):
            return inputs.clone().detach().to(self.device)
        else:
            return torch.tensor(inputs, device=self.device)

    def to_tensors(self, *inputs):
        return [self.to_tensor(inputs_) if inputs_ is not None and inputs_.size
                else None for inputs_ in inputs]

    def optimize(self, loss):
        self.opt.zero_grad()
        loss.backward()
        if self.expe.config.gclip is not None:
            torch.nn.utils.clip_grad_norm(
                self.parameters(), self.expe.config.gclip)
        self.opt.step()

    def init_optimizer(self, opt_type, learning_rate, weight_decay):
        if opt_type.lower() == "adam":
            optimizer = torch.optim.Adam
        elif opt_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop
        elif opt_type.lower() == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError("invalid optimizer: {}".format(opt_type))

        opt = optimizer(
            params=filter(
                lambda p: p.requires_grad, self.parameters()
            ),
            weight_decay=weight_decay,
            lr=learning_rate)

        return opt

    def save(self, dev_bleu, dev_stats, test_bleu, test_stats,
             epoch, iteration=None, name="best"):
        save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
        checkpoint = {
            "dev_bleu": dev_bleu,
            "dev_stats": dev_stats,
            "test_bleu": test_bleu,
            "test_stats": test_stats,
            "epoch": epoch,
            "iteration": iteration,
            "state_dict": self.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "config": self.expe.config
        }
        torch.save(checkpoint, save_path)
        self.expe.log.info("model saved to {}".format(save_path))

    def load(self, checkpointed_state_dict=None, name="best"):
        if checkpointed_state_dict is None:
            save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
            checkpoint = torch.load(save_path,
                                    map_location=lambda storage,
                                    loc: storage)
            self.load_state_dict(checkpoint['state_dict'])
            self.opt.load_state_dict(checkpoint.get("opt_state_dict"))
            self.expe.log.info("model loaded from {}".format(save_path))
            self.to(self.device)
            for state in self.opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.expe.log.info("transferred model to {}".format(self.device))
            return checkpoint.get('epoch', 0), \
                checkpoint.get('iteration', 0), \
                checkpoint.get('dev_bleu', 0), \
                checkpoint.get('test_bleu', 0)
        else:
            self.load_state_dict(checkpointed_state_dict)
            self.expe.log.info("model loaded from checkpoint.")
            self.to(self.device)
            self.expe.log.info("transferred model to {}".format(self.device))


class vgvae(base):
    @auto_init_pytorch
    @auto_init_args
    def __init__(self, vocab_size, embed_dim, embed_init, experiment):
        super(vgvae, self).__init__(
            vocab_size, embed_dim, embed_init, experiment)
        self.yencode = getattr(encoders, self.expe.config.yencoder_type)(
            embed_dim=embed_dim,
            embed_init=embed_init,
            hidden_size=self.expe.config.ensize,
            vocab_size=vocab_size,
            dropout=self.expe.config.dp,
            log=experiment.log)

        self.zencode = getattr(encoders, self.expe.config.zencoder_type)(
            embed_dim=embed_dim,
            embed_init=embed_init,
            hidden_size=self.expe.config.ensize,
            vocab_size=vocab_size,
            dropout=self.expe.config.dp,
            mlp_hidden_size=self.expe.config.mhsize,
            mlp_layer=self.expe.config.mlplayer,
            ncode=self.expe.config.ncode,
            nclass=self.expe.config.nclass,
            log=experiment.log)

        if "lstm" in self.expe.config.yencoder_type.lower():
            y_out_size = 2 * self.expe.config.ensize
        elif self.expe.config.yencoder_type.lower() == "word_avg":
            y_out_size = embed_dim

        if "lstm" in self.expe.config.zencoder_type.lower():
            z_out_size = 2 * self.expe.config.ensize
        elif self.expe.config.zencoder_type.lower() == "word_avg":
            z_out_size = embed_dim

        self.mean1 = model_utils.get_mlp(
            input_size=y_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.ysize,
            n_layer=self.expe.config.ymlplayer,
            dropout=self.expe.config.dp)

        self.logvar1 = model_utils.get_mlp(
            input_size=y_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=1,
            n_layer=self.expe.config.ymlplayer,
            dropout=self.expe.config.dp)

        self.mean2 = model_utils.get_mlp(
            input_size=z_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.zsize,
            n_layer=self.expe.config.zmlplayer,
            dropout=self.expe.config.dp)

        self.logvar2 = model_utils.get_mlp(
            input_size=z_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.zsize,
            n_layer=self.expe.config.zmlplayer,
            dropout=self.expe.config.dp)

        self.decode = getattr(decoders, self.expe.config.decoder_type)(
            embed_init=embed_init,
            embed_dim=embed_dim,
            ysize=self.expe.config.ysize,
            zsize=self.expe.config.zsize,
            mlp_hidden_size=self.expe.config.mhsize,
            mlp_layer=self.expe.config.mlplayer,
            hidden_size=self.expe.config.desize,
            dropout=self.expe.config.dp,
            vocab_size=vocab_size,
            log=experiment.log)

        if "lc" in self.expe.config.zencoder_type.lower():
            enc_embed_dim = embed_dim // self.expe.config.ncode *\
                self.expe.config.ncode
        else:
            enc_embed_dim = embed_dim

        self.enc_pos_decode = model_utils.get_mlp(
            input_size=self.expe.config.zsize + enc_embed_dim,
            hidden_size=self.expe.config.mhsize,
            n_layer=self.expe.config.mlplayer,
            output_size=MAX_LEN,
            dropout=self.expe.config.dp)

        self.dec_pos_decode = model_utils.get_mlp(
            input_size=self.expe.config.zsize + embed_dim,
            hidden_size=self.expe.config.mhsize,
            n_layer=self.expe.config.mlplayer,
            output_size=MAX_LEN,
            dropout=self.expe.config.dp)

    def sent2param(self, sent, sent_repl, mask):
        yembed, yvecs = self.yencode(sent, mask)
        zembed, zvecs = self.zencode(sent_repl, mask)

        mean = self.mean1(yvecs)
        mean = mean / mean.norm(dim=-1, keepdim=True)
        logvar = self.logvar1(yvecs)
        var = F.softplus(logvar) + 1

        mean2 = self.mean2(zvecs)
        logvar2 = self.logvar2(zvecs)

        return zembed, mean, var, mean2, logvar2

    def forward(self, sent1, sent_repl1, mask1, sent2, sent_repl2,
                mask2, tgt1, tgt_mask1, tgt2, tgt_mask2, vtemp, gtemp):
        self.train()
        sent1, sent_repl1, mask1, sent2, sent_repl2, mask2, tgt1, \
            tgt_mask1, tgt2, tgt_mask2 = \
            self.to_tensors(sent1, sent_repl1, mask1, sent2, sent_repl2,
                            mask2, tgt1, tgt_mask1, tgt2, tgt_mask2)

        s1_zembed, sent1_mean, sent1_var, sent1_mean2, sent1_logvar2 = \
            self.sent2param(sent1, sent_repl1, mask1)
        s2_zembed, sent2_mean, sent2_var, sent2_mean2, sent2_logvar2 = \
            self.sent2param(sent2, sent_repl2, mask2)

        sent1_dist = VonMisesFisher(sent1_mean, sent1_var)
        sent2_dist = VonMisesFisher(sent2_mean, sent2_var)

        sent1_syntax = self.sample_gaussian(sent1_mean2, sent1_logvar2)
        sent2_syntax = self.sample_gaussian(sent2_mean2, sent2_logvar2)

        sent1_semantic = sent1_dist.rsample()
        sent2_semantic = sent2_dist.rsample()

        logloss1, s1_decs = self.decode(
            sent1_semantic, sent1_syntax, tgt1, tgt_mask1)
        logloss2, s2_decs = self.decode(
            sent2_semantic, sent2_syntax, tgt2, tgt_mask2)

        logloss3, s3_decs = self.decode(
            sent2_semantic, sent1_syntax, tgt1, tgt_mask1)
        logloss4, s4_decs = self.decode(
            sent1_semantic, sent2_syntax, tgt2, tgt_mask2)

        if self.expe.config.pratio:
            s1_vecs = torch.cat(
                [s1_zembed,
                 sent1_syntax.unsqueeze(1)
                 .expand(-1, s1_zembed.size(1), -1)],
                -1)
            s2_vecs = torch.cat(
                [s2_zembed,
                 sent2_syntax.unsqueeze(1)
                 .expand(-1, s2_zembed.size(1), -1)],
                -1)
            ploss1 = self.pos_loss(
                mask1, s1_vecs, self.enc_pos_decode)
            ploss2 = self.pos_loss(
                mask2, s2_vecs, self.enc_pos_decode)
            ploss3 = self.pos_loss(
                tgt_mask1, s3_decs, self.dec_pos_decode)
            ploss4 = self.pos_loss(
                tgt_mask2, s4_decs, self.dec_pos_decode)
            ploss = ploss1 + ploss2 + ploss3 + ploss4
        else:
            ploss = torch.zeros_like(logloss1)

        sent1_kl = model_utils.gauss_kl_div(
            sent1_mean2, sent1_logvar2,
            eps=self.eps).mean()
        sent2_kl = model_utils.gauss_kl_div(
            sent2_mean2, sent2_logvar2,
            eps=self.eps).mean()

        vkl = sent1_dist.kl_div().mean() + sent2_dist.kl_div().mean()

        gkl = sent1_kl + sent2_kl

        rec_logloss = logloss1 + logloss2

        para_logloss = logloss3 + logloss4

        loss = self.expe.config.lratio * rec_logloss + \
            self.expe.config.plratio * para_logloss + \
            vtemp * vkl + gtemp * gkl + \
            self.expe.config.pratio * ploss

        return loss, vkl, gkl, rec_logloss, para_logloss, ploss

    def greedy_decode(self, semantics, semantics_mask,
                      synatx, syntax_mask, max_len):
        self.eval()
        synatx, syntax_mask, semantics, semantics_mask = \
            self.to_tensors(synatx, syntax_mask, semantics, semantics_mask)
        _, yvecs = self.yencode(semantics, semantics_mask)
        _, zvecs = self.zencode(synatx, syntax_mask)

        ymean = self.mean1(yvecs)
        ymean = ymean / ymean.norm(dim=-1, keepdim=True)

        zmean = self.mean2(zvecs)

        return self.decode.greedy_decode(ymean, zmean, max_len)
