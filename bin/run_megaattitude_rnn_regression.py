import argparse
import numpy as np
import pandas as pd
from torch.nn import LSTM
from torch.cuda import is_available
from torch import device
from factslab.utility import load_glove_embedding, arrange_inputs
from factslab.datastructures import ConstituencyTree
from factslab.pytorch.childsumtreelstm import ChildSumConstituencyTreeLSTM
from factslab.pytorch.rnnregression import RNNRegressionTrainer
import sys
import random

# initialize argument parser
description = 'Run an RNN regression on MegaAttitude.'
parser = argparse.ArgumentParser(description=description)

# file handling
parser.add_argument('--data',
                    type=str,
                    default='../../factslab-data/megaattitude/megaattitude_v1.csv')
parser.add_argument('--structures',
                    type=str,
                    default='../../factslab-data/megaattitude/structures.tsv')
parser.add_argument('--embeddings',
                    type=str,
                    default='../../../embeddings/glove/glove.42B.300d')
parser.add_argument('--regressiontype',
                    type=str,
                    default="linear")
parser.add_argument('--epochs',
                    type=int,
                    default=1)
parser.add_argument('--batch',
                    type=int,
                    default=128)
parser.add_argument('--rnntype',
                    type=str,
                    default="tree")
parser.add_argument('--verbosity',
                    type=int,
                    default="1")
parser.add_argument('--attention',
                    action='store_true',
                    help='Turn attention on or off')

# parse arguments
args = parser.parse_args()

data = pd.read_csv(args.data)

# remove subjects that are marked for exclusion
data = data[~data.exclude]

# remove null responses; removes 10 lines
data = data[~data.response.isnull()]

# the intransitive frame is denoted by an empty string, so make it overt
data.loc[data.frame.isnull(), 'frame'] = 'null'

if args.regressiontype == "multinomial":
    # make smallest response value 0
    data['response'] = data.response.astype(int) - 1

else:
    # convert responses to logit ridit scores
    data['response'] = data.groupby('participant').response.apply(lambda x: x.rank() / (len(x) + 1.))
    data['response'] = np.log(data.response) - np.log(1. - data.response)

# convert "email" to "e-mail" to deal with differences between
# megaattitude_v1.csv and structures.tsv
data['condition'] = data.verb.replace('email', 'e-mail') + '-' + data.frame + '-' + data.voice

# load structures into a dictionary
with open(args.structures) as f:
    structures = dict([line.replace(',', 'COMMA').strip().split('\t') for line in f])

    structures = {k: ConstituencyTree.fromstring(s) for k, s in structures.items()}

for s in structures.values():
    s.collapse_unary(True, True)

# get the structure IDs from the dictionary keys
conditions = list(structures.keys())

# filter down to those conditions found in conditions
data = data[data.condition.isin(conditions)]

# build the vocab list up from the structures
vocab = list({word
              for tree in structures.values()
              for word in tree.leaves()})

# load the glove embedding
embeddings = load_glove_embedding(args.embeddings, vocab)
device_to_use = device("cuda:0" if is_available() else "cpu")
attributes = ['acceptability']

if args.rnntype == "tree":
    rnntype = ChildSumConstituencyTreeLSTM
    x_raw = [structures[c] for c in data.condition.values]
    x = [x_raw[i:i + args.batch] for i in range(0, len(x_raw), args.batch)]
    y_raw = data.response.values
    y = [y_raw[i:i + args.batch] for i in range(0, len(y_raw), args.batch)]

elif args.rnntype == "linear":
    rnntype = LSTM
    x_raw = [structures[c].words() for c in data.condition.values]
    y_raw = data.response.values
    combined = list(zip(x_raw, y_raw))
    random.shuffle(combined)
    x_raw[:], y_raw[:] = zip(*combined)
    dev_x_raw = x_raw[int(len(x_raw) * 0.9):]
    x_raw = x_raw[:int(len(x_raw) * 0.9)]
    dev_y_raw = y_raw[int(len(y_raw) * 0.9):]
    y_raw = y_raw[:int(len(y_raw) * 0.9)]

    x = [x_raw[i:i + args.batch] for i in range(0, len(x_raw), args.batch)]
    dev_x = [dev_x_raw[i:i + args.batch] for i in range(0, len(dev_x_raw), args.batch)]
    x[-1] = x[-1] + x[-2][0:len(x[-2]) - len(x[-1])]
    dev_x[-1] = dev_x[-1] + dev_x[-2][0:len(dev_x[-2]) - len(dev_x[-1])]

    y = {}
    dev_y = {}
    wts_batch = {}
    for attr in attributes:
        y[attr] = [y_raw[i:i + args.batch] for i in range(0, len(y_raw), args.batch)]
        dev_y[attr] = [dev_y_raw[i:i + args.batch] for i in range(0, len(dev_y_raw), args.batch)]
        y[attr][-1] = np.append(y[attr][-1], y[attr][-2][0:len(y[attr][-2]) - len(y[attr][-1])])
        dev_y[attr][-1] = np.append(dev_y[attr][-1], dev_y[attr][-2][0:len(dev_y[attr][-2]) - len(dev_y[attr][-1])])
        wts_batch[attr] = [[None for i in range(args.batch)] for j in range(len(x))]

    tokens_batch = [[None for i in range(args.batch)] for j in range(len(x))]
    x, y, loss_wts, lengths, tokens = arrange_inputs(data_batch=x,
                                                     targets_batch=y,
                                                     wts_batch=wts_batch,
                                                     tokens_batch=tokens_batch,
                                                     attributes=attributes)
    dev_x, dev_y, _, dev_lengths, _ = arrange_inputs(data_batch=dev_x,
                                           targets_batch=dev_y,
                                           wts_batch=wts_batch,
                                           tokens_batch=tokens_batch,
                                           attributes=attributes)
else:
    sys.exit('Error. Argument rnntype must be tree or linear')
import ipdb; ipdb.set_trace()
# train the model
trainer = RNNRegressionTrainer(embeddings=embeddings, device=device_to_use,
                               rnn_classes=rnntype, bidirectional=True,
                               attention=args.attention, epochs=args.epochs,
                               regression_type=args.regressiontype,
                               rnn_hidden_sizes=300, num_rnn_layers=1,
                               regression_hidden_sizes=(150,),
                               batch_size=args.batch, attributes=attributes)

trainer.fit(X=x, Y=y, lr=1e-2, lengths=lengths, verbosity=args.verbosity,
            dev=[dev_x, dev_y, dev_lengths])
