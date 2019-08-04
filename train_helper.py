import time
import logging
import argparse
import os
import torch
import data_utils
import rouge
import signal
import subprocess

import numpy as np

from config import get_base_parser, MULTI_BLEU_PERL, \
    EOS_IDX, MAX_GEN_LEN
from decorators import auto_init_args
from eval_utils import Meteor


def register_exit_handler(exit_handler):
    import atexit

    atexit.register(exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    signal.signal(signal.SIGINT, exit_handler)


def run_multi_bleu(input_file, reference_file):
    bleu_output = subprocess.check_output(
        "./{} -lc {} < {}".format(MULTI_BLEU_PERL, reference_file, input_file),
        stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    bleu = float(
        bleu_output.strip().split("\n")[-1]
        .split(",")[0].split("=")[1][1:])
    return bleu


def get_kl_anneal_function(anneal_function, max_value, slope):
    if anneal_function.lower() == 'exp':
        return lambda step, curr_value: \
            min(max_value, float(1 / (1 + np.exp(-slope * step + 100))))
    elif anneal_function.lower() == 'linear':
        return lambda step, curr_value: \
            min(max_value, curr_value + slope * max_value / (step + 100))
    elif anneal_function.lower() == 'linear2':
        return lambda step, curr_value: min(max_value, slope * step)
    else:
        raise ValueError("invalid anneal function: {}".format(anneal_function))


class tracker:
    @auto_init_args
    def __init__(self, names):
        assert len(names) > 0
        self.reset()

    def __getitem__(self, name):
        return self.values.get(name, 0) / self.counter if self.counter else 0

    def __len__(self):
        return len(self.names)

    def reset(self):
        self.values = dict({name: 0. for name in self.names})
        self.counter = 0
        self.create_time = time.time()

    def update(self, named_values, count):
        """
        named_values: dictionary with each item as name: value
        """
        self.counter += count
        for name, value in named_values.items():
            self.values[name] += value.item() * count

    def summarize(self, output=""):
        if output:
            output += ", "
        for name in self.names:
            output += "{}: {:.3f}, ".format(
                name, self.values[name] / self.counter if self.counter else 0)
        output += "elapsed time: {:.1f}(s)".format(
            time.time() - self.create_time)
        return output

    @property
    def stats(self):
        return {n: v / self.counter if self.counter else 0
                for n, v in self.values.items()}


class experiment:
    @auto_init_args
    def __init__(self, config, experiments_prefix, logfile_name="log"):
        """Create a new Experiment instance.

        Modified based on: https://github.com/ex4sperans/mag

        Args:
            logfile_name: str, naming for log file. This can be useful to
                separate logs for different runs on the same experiment
            experiments_prefix: str, a prefix to the path where
                experiment will be saved
        """

        # get all defaults
        all_defaults = {}
        for key in vars(config):
            all_defaults[key] = get_base_parser().get_default(key)

        self.default_config = all_defaults

        config.resume = False
        if not config.debug:
            if os.path.isdir(self.experiment_dir):
                print("log exists: {}".format(self.experiment_dir))
                config.resume = True

            print(config)
            self._makedir()

        self._make_misc_dir()

    def _makedir(self):
        os.makedirs(self.experiment_dir, exist_ok=True)

    def _make_misc_dir(self):
        os.makedirs(self.config.vocab_file, exist_ok=True)

    @property
    def experiment_dir(self):
        if self.config.debug:
            return "./"
        else:
            # get namespace for each group of args
            arg_g = dict()
            for group in get_base_parser()._action_groups:
                group_d = {a.dest: self.default_config.get(a.dest, None)
                           for a in group._group_actions}
                arg_g[group.title] = argparse.Namespace(**group_d)

            # skip default value
            identifier = ""
            for key, value in sorted(vars(arg_g["model_configs"]).items()):
                if getattr(self.config, key) != value:
                    identifier += key + str(getattr(self.config, key))
            return os.path.join(self.experiments_prefix, identifier)

    @property
    def log_file(self):
        return os.path.join(self.experiment_dir, self.logfile_name)

    def register_directory(self, dirname):
        directory = os.path.join(self.experiment_dir, dirname)
        os.makedirs(directory, exist_ok=True)
        setattr(self, dirname, directory)

    def _register_existing_directories(self):
        for item in os.listdir(self.experiment_dir):
            fullpath = os.path.join(self.experiment_dir, item)
            if os.path.isdir(fullpath):
                setattr(self, item, fullpath)

    def __enter__(self):

        if self.config.debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')
        else:
            print("log saving to", self.log_file)
            logging.basicConfig(
                filename=self.log_file,
                filemode='a+', level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')

        self.log = logging.getLogger()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        logging.shutdown()

    @property
    def elapsed_time(self):
        return (time.time() - self.start_time) / 3600


class evaluator:
    @auto_init_args
    def __init__(self, inp_path, ref_path, model, vocab, inv_vocab, experiment):
        self.expe = experiment
        self.ref_path = ref_path
        self.semantics_input = []
        self.syntax_input = []
        self.references = []
        self.expe.log.info("loading eval input from: {}".format(inp_path))
        with open(inp_path) as fp:
            for line in fp:
                seman_in, syn_in = line.strip().split("\t")
                self.semantics_input.append(
                    [vocab.get(w.lower(), 0) for w in
                     seman_in.strip().split(" ")])
                self.syntax_input.append([vocab.get(w.lower(), 0) for w in
                                          syn_in.strip().split(" ")])

        self.expe.log.info("loading reference from: {}".format(ref_path))
        with open(ref_path) as fp:
            for line in fp:
                self.references.append(line.strip().lower())
        self.expe.log.info("#data: {}, {}, {}".format(
            len(self.semantics_input),
            len(self.syntax_input), len(self.references)))

    def evaluate(self, gen_fn):
        self.model.eval()
        meteor = Meteor()
        rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                 max_n=2,
                                 limit_length=True,
                                 length_limit=MAX_GEN_LEN,
                                 length_limit_type='words',
                                 apply_avg=False,
                                 apply_best=False,
                                 alpha=0.5,  # Default F1_score
                                 weight_factor=1.2,
                                 stemming=True)
        stats = {"bleu": 0, "rouge1": 0,
                 "rouge2": 0, "rougel": 0, "meteor": 0}
        all_gen = []
        for s1, _, m1, s2, _, m2, _, _, _, _, _ in \
            data_utils.minibatcher(
                data1=np.array(self.semantics_input),
                tag1=np.array(self.semantics_input),
                data2=np.array(self.syntax_input),
                tag2=np.array(self.syntax_input),
                tag_bucket=None,
                batch_size=100,
                p_replace=0.,
                shuffle=False,
                p_scramble=0.):
            with torch.no_grad():
                batch_gen = self.model.greedy_decode(
                    s1, m1, s2, m2, MAX_GEN_LEN)
            for gen in batch_gen:
                curr_gen = []
                for i in gen:
                    if i == EOS_IDX:
                        break
                    curr_gen.append(self.inv_vocab[int(i)])
                all_gen.append(" ".join(curr_gen))
        assert len(all_gen) == len(self.references), \
            "{} != {}".format(len(all_gen), len(self.references))
        fn = os.path.join(self.expe.experiment_dir, gen_fn + ".txt")
        with open(fn, "w+") as fp:
            for hyp, ref in zip(all_gen, self.references):
                fp.write(hyp)
                fp.write("\n")
                try:
                    stats['meteor'] += meteor._score(hyp, [ref])
                except ValueError:
                    stats['meteor'] += 0
                rs = rouge_eval.get_scores([hyp], [ref])
                stats['rouge1'] += rs['rouge-1'][0]['f'][0]
                stats['rouge2'] += rs['rouge-2'][0]['f'][0]
                stats['rougel'] += rs['rouge-l'][0]['f'][0]

        stats['bleu'] = run_multi_bleu(fn, self.ref_path)
        self.expe.log.info("generated sentences saved to: {}".format(fn))

        stats['meteor'] = stats['meteor'] / len(all_gen) * 100
        stats['rouge1'] = stats['rouge1'] / len(all_gen) * 100
        stats['rouge2'] = stats['rouge2'] / len(all_gen) * 100
        stats['rougel'] = stats['rougel'] / len(all_gen) * 100

        self.expe.log.info(
            "#Data: {}, bleu: {:.3f}, meteor: {:.3f}, "
            "rouge-1: {:.3f}, rouge-2: {:.3f}, rouge-l: {:.3f}"
            .format(len(all_gen), stats['bleu'], stats['meteor'],
                    stats['rouge1'], stats['rouge2'], stats['rougel']))
        return stats, stats['bleu']
