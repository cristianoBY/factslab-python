import argparse
import numpy as np
import pandas as pd

from torch.nn import LSTM, SmoothL1Loss, L1Loss, CrossEntropyLoss
from torch.cuda import is_available
from factslab.utility import load_glove_embedding
from factslab.datastructures import ConstituencyTree
from factslab.pytorch.childsumtreelstm import ChildSumConstituencyTreeLSTM
from factslab.pytorch.rnnregression import RNNRegressionTrainer

## initialize argument parser
description = 'Run an RNN regression on MegaAttitude.'
parser = argparse.ArgumentParser(description=description)

## file handling
parser.add_argument('--data', 
                    type=str, 
                    default='../../factslab-data/megaattitude/megaattitude_v1.csv')
parser.add_argument('--structures', 
                    type=str, 
                    default='../../factslab-data/megaattitude/structures.tsv')
parser.add_argument('--regressiontype', 
                    type=str, 
                    default="linear")

## parse arguments
args = parser.parse_args()

data = pd.read_csv(args.data)

# remove subjects that are marked for exclusion
data = data[~data.exclude]

# remove null responses; removes 10 lines
data = data[~data.response.isnull()]

# the intransitive frame is denoted by an empty string, so make it overt
data.loc[data.frame.isnull(),'frame'] = 'null'

if args.regressiontype == "multinomial":
    # make smallest response value 0
    data['response'] = data.response.astype(int) - 1
    
else:
    # convert responses to logit ridit scores
    data['response'] = data.groupby('participant').response.apply(lambda x: x.rank()/(len(x)+1.))
    data['response'] = np.log(data.response) - np.log(1.-data.response)
    
# convert "email" to "e-mail" to deal with differences between
# megaattitude_v1.csv and structures.tsv
data['condition'] = data.verb.replace('email', 'e-mail')+'-'+data.frame+'-'+data.voice

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
embeddings = load_glove_embedding('../../../embeddings/glove/glove.42B.300d', vocab)

cuda_or_not = is_available()

# train the model
trainer = RNNRegressionTrainer(embeddings=embeddings, gpu=cuda_or_not,
                               rnn_classes=ChildSumConstituencyTreeLSTM,
                               bidirectional=True, attention=True,
                               regression_type=args.regressiontype,
                               rnn_hidden_sizes=300, num_rnn_layers=1,
                               regression_hidden_sizes=(150,))
trainer.fit(X=[[structures[c] for c in data.condition.values]],
            Y=data.response.values,
            lr=1e-2, batch_size=10,
            verbosity=10)

# trainer = RNNRegressionTrainer(embeddings=embeddings, gpu=True,
#                                rnn_classes=[LSTM, ChildSumConstituencyTreeLSTM],
#                                regression_type=args.regressiontype,
#                                rnn_hidden_sizes=300, num_rnn_layers=1,
#                                regression_hidden_sizes=(150,))
# trainer.fit(X=[[structures[c].words() for c in data.condition.values],
#                [structures[c] for c in data.condition.values]],
#             Y=data.response.values,
#             lr=1., batch_size=1000)
