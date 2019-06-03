import os
import pickle

import numpy as np

from collections import Counter

from decorators import auto_init_args, lazy_execute
from config import UNK_IDX, UNK_WORD, BOS_IDX, EOS_IDX


class data_holder:
    @auto_init_args
    def __init__(self, train_data, train_tag, vocab, tag_bucket):
        self.inv_vocab = {i: w for w, i in vocab.items()}


class data_processor:
    @auto_init_args
    def __init__(self, train_path, experiment):
        self.expe = experiment

    def process(self):
        if self.expe.config.pre_train_emb:
            fn = "pre_vocab_" + str(self.expe.config.vocab_size)
        else:
            fn = "vocab_" + str(self.expe.config.vocab_size)

        vocab_file = os.path.join(self.expe.config.vocab_file, fn)

        train_data = self._load_sent(
            self.train_path, file_name=self.train_path + ".pkl")

        if self.expe.config.pre_train_emb:
            W, vocab = \
                self._build_pretrain_vocab(train_data, file_name=vocab_file)
        else:
            W, vocab = \
                self._build_vocab(train_data, file_name=vocab_file)
        self.expe.log.info("vocab size: {}".format(len(vocab)))

        self.expe.log.info("initializing pos bucketing")
        tag_bucket = self._load_tag_bucket(
            vocab, self.expe.config.tag_vocab_file,
            file_name=vocab_file + "_tag")

        train_tag1 = []
        train_tag2 = []
        with open(self.expe.config.train_tag_path) as fp:
            for line in fp:
                s1, s2 = line.strip().split("\t")
                train_tag1.append(s1.split(" "))
                train_tag2.append(s2.split(" "))
        assert len(train_tag1) == len(train_data[0])
        assert len(train_tag2) == len(train_data[1])
        train_tag = [np.array(train_tag1), np.array(train_tag2)]

        train_data = self._data_to_idx(train_data, vocab)

        def cal_stats(data):
            unk_count = 0
            total_count = 0
            leng = []
            for sent1, sent2 in zip(*data):
                leng.append(len(sent1))
                leng.append(len(sent2))
                for w in sent1 + sent2:
                    if w == UNK_IDX:
                        unk_count += 1
                    total_count += 1
            return (unk_count, total_count, unk_count / total_count), \
                (len(leng), max(leng), min(leng), sum(leng) / len(leng))

        train_unk_stats, train_len_stats = cal_stats(train_data)
        self.expe.log.info("#train data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}"
                           .format(*train_len_stats))

        self.expe.log.info("#unk in train sentences: {}"
                           .format(train_unk_stats))
        data = data_holder(
            train_data=train_data,
            train_tag=train_tag,
            vocab=vocab,
            tag_bucket=tag_bucket)

        return data, W

    @lazy_execute("_load_from_pickle")
    def _load_tag_bucket(self, vocab, file_path):
        with open(file_path, "rb") as fp:
            word2tag = pickle.load(fp)
        tag2vocab = {}
        for w, tags in word2tag.items():
            if w in vocab:
                for tag in tags.most_common(2):
                    if tag[0] not in tag2vocab:
                        tag2vocab[tag[0]] = [vocab[w]]
                    else:
                        tag2vocab[tag[0]].append(vocab[w])
        self.expe.log.info("#tags: {}".format(len(tag2vocab)))
        return tag2vocab

    @lazy_execute("_load_from_pickle")
    def _load_sent(self, path):
        data_pair1 = []
        data_pair2 = []
        with open(path) as f:
            for line in f:
                line = line.strip().lower()
                if len(line):
                    line = line.split('\t')
                    if len(line) == 2:
                        data_pair1.append(line[0].split(" "))
                        data_pair2.append(line[1].split(" "))
                    else:
                        self.expe.log.warning("unexpected data: " + line)
        assert len(data_pair1) == len(data_pair2)
        return data_pair1, data_pair2

    def _data_to_idx(self, data, vocab):
        idx_pair1 = []
        idx_pair2 = []
        for d1, d2 in zip(*data):
            s1 = [vocab.get(w, UNK_IDX) for w in d1]
            idx_pair1.append(s1)
            s2 = [vocab.get(w, UNK_IDX) for w in d2]
            idx_pair2.append(s2)
        return np.array(idx_pair1), np.array(idx_pair2)

    def _load_paragram_embedding(self, path):
        with open(path, encoding="latin-1") as fp:
            # word_vectors: word --> vector
            word_vectors = {}
            for line in fp:
                line = line.strip("\n").split(" ")
                word_vectors[line[0]] = np.array(
                    list(map(float, line[1:])), dtype='float32')
        vocab_embed = word_vectors.keys()
        embed_dim = word_vectors[next(iter(vocab_embed))].shape[0]
        return word_vectors, vocab_embed, embed_dim

    def _load_glove_embedding(self, path):
        with open(path, 'r', encoding='utf8') as fp:
            # word_vectors: word --> vector
            word_vectors = {}
            for line in fp:
                line = line.strip("\n").split(" ")
                word_vectors[line[0]] = np.array(
                    list(map(float, line[1:])), dtype='float32')
        vocab_embed = word_vectors.keys()
        embed_dim = word_vectors[next(iter(vocab_embed))].shape[0]

        return word_vectors, vocab_embed, embed_dim

    def _create_vocab_from_data(self, data):
        vocab = Counter()
        for sent1, sent2 in zip(*data):
            for w in sent1 + sent2:
                vocab[w] += 1

        ls = vocab.most_common(self.expe.config.vocab_size)
        self.expe.log.info(
            '#Words: %d -> %d' % (len(vocab), len(ls)))
        for key in ls[:5]:
            self.expe.log.info(key)
        self.expe.log.info('...')
        for key in ls[-5:]:
            self.expe.log.info(key)
        vocab = [x[0] for x in ls]

        # 0: unk, 1: bos, 2: eos
        vocab = {w: index + 3 for (index, w) in enumerate(vocab)}
        vocab[UNK_WORD] = UNK_IDX
        vocab["<bos>"] = BOS_IDX
        vocab["<eos>"] = EOS_IDX

        return vocab

    @lazy_execute("_load_from_pickle")
    def _build_vocab(self, train_data):
        vocab = self._create_vocab_from_data(train_data)
        return None, vocab

    @lazy_execute("_load_from_pickle")
    def _build_pretrain_vocab(self, train_data):
        self.expe.log.info("loading embedding from: {}"
                           .format(self.expe.config.embed_file))
        if self.expe.config.embed_type.lower() == "glove":
            word_vectors, vocab_embed, embed_dim = \
                self._load_glove_embedding(self.expe.config.embed_file)
        elif self.expe.config.embed_type.lower() == "paragram":
            word_vectors, vocab_embed, embed_dim = \
                self._load_paragram_embedding(self.expe.config.embed_file)
        else:
            raise NotImplementedError(
                "invalid embedding type: {}".format(
                    self.expe.config.embed_type))

        vocab = self._create_vocab_from_data(train_data)

        W = np.random.uniform(
            -np.sqrt(3.0 / embed_dim), np.sqrt(3.0 / embed_dim),
            size=(len(vocab), embed_dim)).astype('float32')
        n = 0
        for w, i in vocab.items():
            if w in vocab_embed:
                W[i, :] = word_vectors[w]
                n += 1
        self.expe.log.info(
            "{}/{} vocabs are initialized with {} embeddings."
            .format(n, len(vocab), self.expe.config.embed_type))

        return W, vocab

    def _load_from_pickle(self, file_name):
        self.expe.log.info("loading from {}".format(file_name))
        with open(file_name, "rb") as fp:
            data = pickle.load(fp)
        return data


class minibatcher:
    @auto_init_args
    def __init__(self, data1, tag1, data2, tag2, tag_bucket,
                 batch_size, shuffle, p_scramble,
                 p_replace, *args, **kwargs):
        self._reset()

    def __len__(self):
        return len(self.idx_pool)

    def _reset(self):
        self.pointer = 0
        idx_list = np.arange(len(self.data1))
        if self.shuffle:
            np.random.shuffle(idx_list)
        self.idx_pool = [idx_list[i: i + self.batch_size]
                         for i in range(0, len(self.data1), self.batch_size)]

    def _replace_word(self, sent, tag):
        assert len(sent) == len(tag)
        new_sent = []
        for w, t in zip(sent, tag):
            if np.random.choice(
                    [True, False],
                    p=[self.p_replace, 1 - self.p_replace]).item():
                new_sent.append(np.random.choice(self.tag_bucket[t]))
            else:
                new_sent.append(w)
        return new_sent

    def _pad(self, data1, tag1, data2, tag2):
        assert len(data1) == len(data2)
        max_len1 = max([len(sent) for sent in data1])
        max_len2 = max([len(sent) for sent in data2])

        input_data1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        input_repl_data1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        input_mask1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        tgt_data1 = \
            np.zeros((len(data1), max_len1 + 2)).astype("float32")
        tgt_mask1 = \
            np.zeros((len(data1), max_len1 + 2)).astype("float32")

        input_data2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        input_repl_data2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        input_mask2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        tgt_data2 = \
            np.zeros((len(data2), max_len2 + 2)).astype("float32")
        tgt_mask2 = \
            np.zeros((len(data2), max_len2 + 2)).astype("float32")

        for i, (sent1, t1, sent2, t2) in \
                enumerate(zip(data1, tag1, data2, tag2)):
            if np.random.choice(
                    [True, False],
                    p=[self.p_scramble, 1 - self.p_scramble]).item():
                sent1 = np.random.permutation(sent1)
                sent2 = np.random.permutation(sent2)

            input_data1[i, :len(sent1)] = \
                np.asarray(list(sent1)).astype("float32")
            input_mask1[i, :len(sent1)] = 1.

            tgt_data1[i, :len(sent1) + 2] = \
                np.asarray([BOS_IDX] + list(sent1) + [EOS_IDX]).astype("float32")
            tgt_mask1[i, :len(sent1) + 2] = 1.

            input_data2[i, :len(sent2)] = \
                np.asarray(list(sent2)).astype("float32")
            input_mask2[i, :len(sent2)] = 1.

            tgt_data2[i, :len(sent2) + 2] = \
                np.asarray([BOS_IDX] + list(sent2) + [EOS_IDX]).astype("float32")
            tgt_mask2[i, :len(sent2) + 2] = 1.

            input_repl_data1[i, :len(sent1)] = \
                np.asarray(self._replace_word(sent1, t1)).astype("float32")
            input_repl_data2[i, :len(sent2)] = \
                np.asarray(self._replace_word(sent2, t2)).astype("float32")

        return [input_data1, input_repl_data1, input_mask1,
                input_data2, input_repl_data2, input_mask2,
                tgt_data1, tgt_mask1, tgt_data2, tgt_mask2]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            raise StopIteration()

        idx = self.idx_pool[self.pointer]
        data1, data2 = self.data1[idx], self.data2[idx]
        t1, t2 = self.tag1[idx], self.tag2[idx]
        self.pointer += 1
        return self._pad(data1, t1, data2, t2) + [idx]
