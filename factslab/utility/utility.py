import os
import numpy as np
import pandas as pd
from zipfile import ZipFile


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
        oov = pd.DataFrame(np.tile(mean_emb, [len(oov), 1]), index=oov)
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
