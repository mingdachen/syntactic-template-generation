import argparse

UNK_IDX = 0
UNK_WORD = "UUUNKKK"
EVAL_YEAR = "2017"
BOS_IDX = 1
EOS_IDX = 2
MAX_GEN_LEN = 40
METEOR_JAR = 'evaluation/meteor-1.5.jar'
METEOR_DATA = 'evaluation/data/paraphrase-en.gz'
MULTI_BLEU_PERL = 'evaluation/multi-bleu.perl'
STANFORD_CORENLP = 'evaluation/stanford-corenlp-full-2018-10-05'


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_base_parser():
    parser = argparse.ArgumentParser(
        description='Controllable Paraphrase Generation using PyTorch')
    parser.register('type', 'bool', str2bool)

    basic_group = parser.add_argument_group('basics')
    # Basics
    basic_group.add_argument('--debug', type="bool", default=False,
                             help='activation of debug mode (default: False)')
    basic_group.add_argument('--auto_disconnect', type="bool", default=True,
                             help='for slurm (default: True)')
    basic_group.add_argument('--save_prefix', type=str, default="experiments",
                             help='saving path prefix')

    data_group = parser.add_argument_group('data')
    # Data file
    data_group.add_argument('--train_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--train_tag_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--vocab_file', type=str, default=None,
                            help='vocabulary file')
    data_group.add_argument('--tag_vocab_file', type=str, default=None,
                            help='tag vocabulary file')
    data_group.add_argument('--embed_file', type=str, default=None,
                            help='pretrained embedding file')
    data_group.add_argument('--dev_inp_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--dev_ref_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--test_inp_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--test_ref_path', type=str, default=None,
                            help='data file')

    config_group = parser.add_argument_group('model_configs')
    config_group.add_argument('-lr', '--learning_rate',
                              dest='lr',
                              type=float,
                              default=1e-3,
                              help='learning rate')
    config_group.add_argument('-pratio', '--ploss_ratio',
                              dest='pratio',
                              type=float,
                              default=1.0,
                              help='ratio of position loss')
    config_group.add_argument('-lratio', '--logloss_ratio',
                              dest='lratio',
                              type=float,
                              default=1.0,
                              help='ratio of log loss')
    config_group.add_argument('-plratio', '--para_logloss_ratio',
                              dest='plratio',
                              type=float,
                              default=1.0,
                              help='ratio of paraphrase log loss')
    config_group.add_argument('--eps',
                              type=float,
                              default=1e-4,
                              help='safty for avoiding numerical issues')
    config_group.add_argument('-edim', '--embed_dim',
                              dest='edim',
                              type=int, default=300,
                              help='size of embedding')
    config_group.add_argument('-wr', '--word_replace',
                              dest='wr',
                              type=float, default=0.0,
                              help='word replace rate')
    config_group.add_argument('-dp', '--dropout',
                              dest='dp',
                              type=float, default=0.0,
                              help='dropout probability')
    config_group.add_argument('-gclip', '--grad_clip',
                              dest='gclip',
                              type=float, default=None,
                              help='gradient clipping threshold')

    # recurrent neural network detail
    config_group.add_argument('-ensize', '--encoder_size',
                              dest='ensize',
                              type=int, default=300,
                              help='encoder hidden size')
    config_group.add_argument('-desize', '--decoder_size',
                              dest='desize',
                              type=int, default=300,
                              help='decoder hidden size')
    config_group.add_argument('--ysize',
                              dest='ysize',
                              type=int, default=100,
                              help='size of Gaussian')
    config_group.add_argument('--zsize',
                              dest='zsize',
                              type=int, default=100,
                              help='size of Gaussian')

    # feedforward neural network
    config_group.add_argument('-mhsize', '--mlp_hidden_size',
                              dest='mhsize',
                              type=int, default=100,
                              help='size of hidden size')
    config_group.add_argument('-mlplayer', '--mlp_n_layer',
                              dest='mlplayer',
                              type=int, default=3,
                              help='number of layer')
    config_group.add_argument('-zmlplayer', '--zmlp_n_layer',
                              dest='zmlplayer',
                              type=int, default=3,
                              help='number of layer')
    config_group.add_argument('-ymlplayer', '--ymlp_n_layer',
                              dest='ymlplayer',
                              type=int, default=3,
                              help='number of layer')

    # latent code
    config_group.add_argument('-ncode', '--num_code',
                              dest='ncode',
                              type=int, default=8,
                              help='number of latent code')
    config_group.add_argument('-nclass', '--num_class',
                              dest='nclass',
                              type=int, default=2,
                              help='size of classes in each latent code')
    # optimization
    config_group.add_argument('-ps', '--p_scramble',
                              dest='ps',
                              type=float, default=0.,
                              help='probability of scrambling')
    config_group.add_argument('--l2', type=float, default=0.,
                              help='l2 regularization')
    config_group.add_argument('-vmkl', '--max_vmf_kl_temp',
                              dest='vmkl', type=float, default=1.,
                              help='maximum temperature of kl divergence')
    config_group.add_argument('-gmkl', '--max_gauss_kl_temp',
                              dest='gmkl', type=float, default=1.,
                              help='maximum temperature of kl divergence')

    setup_group = parser.add_argument_group('train_setup')
    # train detail
    setup_group.add_argument('--save_dir', type=str, default=None,
                             help='model save path')
    basic_group.add_argument('--embed_type',
                             type=str, default="paragram",
                             choices=['paragram', 'glove'],
                             help='types of embedding: paragram, glove')
    basic_group.add_argument('--yencoder_type',
                             type=str, default="word_avg",
                             help='types of encoder')
    basic_group.add_argument('--zencoder_type',
                             type=str, default="word_avg",
                             help='types of z encoder')
    basic_group.add_argument('--decoder_type',
                             type=str, default="lstm_z2y",
                             help='types of decoder')
    setup_group.add_argument('--n_epoch', type=int, default=5,
                             help='number of epochs')
    setup_group.add_argument('--batch_size', type=int, default=20,
                             help='batch size')
    setup_group.add_argument('--opt', type=str, default='adam',
                             choices=['sadam', 'adam', 'sgd', 'rmsprop'],
                             help='types of optimizer: adam (default), \
                             sgd, rmsprop')
    setup_group.add_argument('--pre_train_emb', type="bool", default=False,
                             help='whether to use pretrain embedding')
    setup_group.add_argument('--vocab_size', type=int, default=50000,
                             help='size of vocabulary')

    misc_group = parser.add_argument_group('misc')
    # misc
    misc_group.add_argument('--print_every', type=int, default=10,
                            help='print training details after \
                            this number of iterations')
    misc_group.add_argument('--eval_every', type=int, default=100,
                            help='evaluate model after \
                            this number of iterations')
    misc_group.add_argument('--summarize', type="bool", default=False,
                            help='whether to summarize training stats\
                            (default: False)')
    return parser
