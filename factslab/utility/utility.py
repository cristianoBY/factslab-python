import os
import numpy as np
import pandas as pd
from zipfile import ZipFile
from torch import from_numpy, sort


def load_glove_embedding(fpath, vocab):
    """load glove embedding

    Parameters
    ----------
    fpath : str
        path to zip containing glove embeddings
    vocab : list(str)
        list of vocab elements to extract from glove
    """

    zipname = os.path.split(fpath)[-1]
    size, dim = zipname.split('.')[1:3]
    fpathout = 'glove.' + size + '.' + dim + '.filtered.txt'
    if fpathout in os.listdir(os.getcwd()):
        embedding = pd.read_csv(fpathout, index_col=0, header=None, sep=' ')

    else:
        fname = 'glove.' + size + '.' + dim + '.txt'
        f_emb = ZipFile(fpath + '.zip').open(fname)

        emb = np.array([l.decode().strip().split()
                        for l in f_emb
                        if l.split()[0].decode() in vocab])

        embedding = pd.DataFrame(emb[:, 1:].astype(float), index=emb[:, 0])

        mean_emb = list(embedding.mean(axis=0).values)
        oov = [w for w in vocab if w not in embedding.index.values]

        # Add an embedding element for padding
        PADDING_ELEMENT = ["<PAD>"]
        oov = pd.DataFrame(np.tile(mean_emb, [len(oov), 1]), index=oov)
        pad = pd.DataFrame([np.zeros(len(mean_emb))], index=PADDING_ELEMENT)
        oov = pd.concat([oov, pad], axis=0)

        embedding = pd.concat([embedding, oov], axis=0)
        embedding.to_csv(fpathout, sep=' ', header=False)

    return embedding


def partition(l, n):
    """partition a list in n blocks"""

    for i in range(0, len(l), n):
        if i < (len(l) - n):
            yield l[i:(i + n)]
        else:
            yield l[i:]


def arrange_inputs(data_batch, targets_batch, wts_batch, tokens_batch, attributes):
        """
            Arrange input sequences so that each minibatch has same length
        """
        sorted_data_batch = []
        sorted_seq_len_batch = []
        sorted_tokens_batch = []
        sorted_idx_batch = []

        sorted_targets_batch = {}
        sorted_wts_batch = {}
        for attr in attributes:
            sorted_targets_batch[attr] = []
            sorted_wts_batch[attr] = []

        for data, tokens in zip(data_batch, tokens_batch):
            seq_len = from_numpy(np.array([len(x) for x in data]))
            sorted_seq_len, sorted_idx = sort(seq_len, descending=True)
            max_len = sorted_seq_len[0]
            sorted_seq_len_batch.append(np.array(sorted_seq_len))
            sorted_data = [data[x] + ['<PAD>' for i in range(max_len - len(data[x]))] for x in sorted_idx]
            if tokens[0] is not None:
                sorted_tokens = np.array([(tokens[x] + 1) for x in sorted_idx])
                sorted_tokens_batch.append(sorted_tokens)
            else:
                sorted_tokens_batch = None
            sorted_data_batch.append(sorted_data)
            sorted_idx_batch.append(sorted_idx)

        for attr in attributes:
            for targets, wts, sorted_idx in zip(targets_batch[attr], wts_batch[attr], sorted_idx_batch):
                sorted_targets = [targets[x] for x in sorted_idx]
                if wts[0] is not None:
                    sorted_wts = [wts[x] for x in sorted_idx]
                    sorted_wts_batch[attr].append(sorted_wts)
                else:
                    sorted_wts_batch = None
                sorted_targets_batch[attr].append(sorted_targets)

        return sorted_data_batch, sorted_targets_batch, sorted_wts_batch, sorted_seq_len_batch, sorted_tokens_batch
