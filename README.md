# syntactic-template-generation
A PyTorch implementation of [Controllable Paraphrase Generation with a Syntactic Exemplar](https://ttic.uchicago.edu/~mchen/papers/mchen+etal.acl19.pdf)

## Requirements

- Python 3.5
- PyTorch >= 1.0
- NLTK
- [tqdm](https://github.com/tqdm/tqdm)
- [py-rouge](https://github.com/Diego999/py-rouge)
- [zss](https://github.com/timtadh/zhang-shasha)

## Resource

- [data and tags](https://drive.google.com/open?id=1HHDlUT_-WpedL6zNYpcN94cLwed_yyrP)
- [evaluation (including multi-bleu, METEOR and a copy of Stanford CoreNLP)](https://drive.google.com/drive/folders/1FJjvMldeZrJnQd-iVXJ3KGFBLEvsndNY?usp=sharing)
- [syntactic evaluation](https://drive.google.com/drive/folders/1oVjn_3xIDZbkRm50fSHDZ5nKZtJ_BFyD?usp=sharing)
- [pretrained model (VGVAE+LC+WN+WPL)](https://drive.google.com/drive/folders/13pii_XG-szMG2KNSuyDn7iPFDyhnXjXm?usp=sharing)

``run_vgvae.sh`` is provided as an example for training new models.

## Generation

#### Generate sentences using beam search (and evaluation)

``python generate.py -s PATH_TO_MODEL_PICKLE -v VOCAB_PICKLE -i SYNTACTIC_SEMANTIC_TEMPLATES -r REFERENCE_FILE -bs BEAM_SIZE``

The argument ``-r`` is optional. When it is specified, the following evaluation script will be executed for reporting BLUE, ROUGE-{1,2,L}, METEOR and Syntactic TED scores.

## Evaluation

#### BLUE, ROUGE, METEOR and Syntactic TED scores
``python eval.py -i INPUT_FILE -r REFERENCE_FILE ``

#### Labeled F1 and Tagging accuracy
``python eval_f1_acc.py -s PATH_TO_MODEL_PICKLE -v VOCAB_PICKLE -d SYNTACTIC_EVAL_DIR``

#### STS benchmark
``python eval_sts.py -s PATH_TO_MODEL_PICKLE -v VOCAB_PICKLE -d PATH_TO_STS``


## Reference
```
@inproceedings{mchen-controllable-19,
  author    = {Mingda Chen and Qingming Tang and Sam Wiseman and Kevin Gimpel},
  title     = {Controllable Paraphrase Generation with a Syntactic Exemplar},
  booktitle = {Proc. of {ACL}},
  year      = {2019}
}
```
