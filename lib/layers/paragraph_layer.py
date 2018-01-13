# --------------------------------------------------------
# Im2P-Tensorflow
# Written by InnerPeace
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""This python layer accepts region ids as input and
retrieves region sentense for them."""

from six.moves import cPickle
from lib.config import cfg
from collections import Counter
import numpy as np
import six
from six.moves import xrange


def paragraph_layer(gt_ptokens):
    assert gt_ptokens.ndim == 2

    num_sentences, n_w = gt_ptokens.shape

    assert n_w == cfg.MAX_WORDS

    if num_sentences > cfg.IM2P.S_MAX:
        num_sentences = cfg.IM2P.S_MAX
        gt_ptokens = gt_ptokens[: num_sentences]
    # else:
    #     repeats = [1] * (num_sentences - 1) + [cfg.IM2P.S_MAX - num_sentences + 1]
    #     gt_ptokens = np.repeat(gt_ptokens, repeats, axis=0)

    sentence_labels = np.array([0] * (num_sentences - 1) + [1], dtype=np.int32)
    #* (cfg.IM2P.S_MAX - num_sentences + 1), dtype=np.int32)
    target_sentences = np.zeros((num_sentences, cfg.TIME_STEPS), dtype=np.float32)
    input_sentences = np.zeros((num_sentences, cfg.TIME_STEPS - 1), dtype=np.float32)
    # add start token "1"
    target_sentences[:, 0] = 1
    input_sentences[:, 0] = 1
    for i in xrange(num_sentences):
        s = gt_ptokens[i]
        target_sentences[i, 1: -1] = s
        # "2" is end of sentence token
        target_sentences[i, np.sum(s > 0) + 1] = 2
        input_sentences[i, 1:] = s

    sentence_lengths = np.sum(target_sentences > 0, axis=1, dtype=np.int32)

    return np.array(num_sentences, dtype=np.int32), sentence_lengths, sentence_labels, input_sentences, target_sentences


if __name__ == '__main__':
    gt_ptokens = np.array([[5, 425, 521, 835, 9, 5, 260, 9, 47, 7, 5, 40, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

    print(paragraph_layer(gt_ptokens))
