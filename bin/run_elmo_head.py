import argparse
import numpy as np
import pandas as pd
from factslab.utility import ridit, dev_mode_group
from factslab.pytorch.mlpregression import MLPTrainer
from torch.cuda import is_available
from torch import device
# from allennlp.modules.elmo import Elmo
from allennlp.commands.elmo import ElmoEmbedder
from sklearn.utils import shuffle
from os.path import expanduser

pd.set_option('display.max_columns', None)

if __name__ == "__main__":
    home = expanduser('~')
    # initialize argument parser
    description = 'Run a simple MLP with(out) attention of varying types on ELMO.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--protocol',
                        type=str,
                        default="noun")
    parser.add_argument('--structures',
                        type=str,
                        default=home + '/Desktop/protocols/data/structures.tsv')
    parser.add_argument('--datapath',
                        type=str,
                        default=home + '/Desktop/protocols/data/')
    parser.add_argument('--embeddings',
                        type=str,
                        default=home + '/Downloads/embeddings/')
    parser.add_argument('--regressiontype',
                        type=str,
                        default="linear")
    parser.add_argument('--epochs',
                        type=int,
                        default=1)
    parser.add_argument('--batch',
                        type=int,
                        default=128)
    parser.add_argument('--verbosity',
                        type=int,
                        default="1")
    parser.add_argument('--span',
                        action='store_true',
                        help='Turn span attention on or off')
    parser.add_argument('--sentence',
                        action='store_true',
                        help='Turn sentence attention on or off')
    parser.add_argument('--param',
                        action='store_true',
                        help='Turn param attention on or off')

    # parse arguments
    args = parser.parse_args()

    # Find out the attention type to be used based on arguments
    if not args.span and not args.sentence:
        attention_type = "None"
    else:
        if not args.param:
            if args.span and not args.sentence:
                attention_type = "Span"
            elif args.sentence and not args.span:
                attention_type = "Sentence"
        else:
            if args.span and not args.sentence:
                attention_type = "Span-param"
            elif args.sentence and not args.span:
                attention_type = "Sentence-param"

    if args.protocol == "noun":
        datafile = args.datapath + "noun_long_data.tsv"
        response = ["Is.Particular", "Is.Kind", "Is.Abstract"]
        response_conf = ["Part.Confidence", "Kind.Confidence", "Abs.Confidence"]
        attributes = ["part", "kind", "abs"]
        attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
        attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
                 "abs": "Abs.Confidence"}
        token_col = "Noun.Token"
        root_token = "Noun.Root.Token"
        dev_cols = ['Unique.ID', token_col, 'Is.Particular.norm', 'Is.Kind.norm', 'Is.Abstract.norm', 'Part.Confidence.norm', 'Kind.Confidence.norm', 'Abs.Confidence.norm']
    else:
        datafile = args.datapath + "pred_long_data.tsv"
        response = ["Is.Particular", "Is.Hypothetical", "Is.Dynamic"]
        response_conf = ["Part.Confidence", "Hyp.Confidence", "Dyn.Confidence"]
        attributes = ["part", "hyp", "dyn"]
        attr_map = {"part": "Is.Particular", "dyn": "Is.Dynamic", "hyp": "Is.Hypothetical"}
        attr_conf = {"part": "Part.Confidence", "dyn": "Dyn.Confidence",
                 "hyp": "Hyp.Confidence"}
        token_col = "Pred.Root.Token"
        root_token = "Pred.Root.Token"
        dev_cols = ['Unique.ID', token_col, 'Is.Particular.norm', 'Is.Dynamic.norm', 'Is.Hypothetical.norm', 'Part.Confidence.norm', 'Dyn.Confidence.norm', 'Hyp.Confidence.norm']

    data = pd.read_csv(datafile, sep="\t")

    data['Split.Sentence.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']), axis=1)

    data['Unique.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']) + "_" + str(x[token_col]), axis=1)

    # Load the structures
    structures = {}

    # Don't read_csv the structures file. read_csv can't handle quotes
    with open(args.structures, 'r') as f:
        for line in f.readlines():
            structs = line.split('\t')
            structures[structs[0]] = structs[1].split()

    data['Structure'] = data['Split.Sentence.ID'].map(lambda x: structures[x])

    # Split the datasets into train, dev, test
    data_test = data[data['Split'] == 'test']
    data_dev = data[data['Split'] == 'dev']
    data = data[data['Split'] == 'train']

    # Ridit scoring annotations and confidence ratings
    for attr in attributes:
        resp = attr_map[attr]
        resp_conf = attr_conf[attr]
        data[resp_conf + ".norm"] = data.groupby('Annotator.ID')[resp_conf].transform(ridit)
        data_dev[resp_conf + ".norm"] = data_dev.groupby('Annotator.ID')[resp_conf].transform(ridit)
        if args.regressiontype == "multinomial":
            data[resp + ".norm"] = data[resp].map(lambda x: 1 if x else 0)
            data_dev[resp + ".norm"] = data_dev[resp].map(lambda x: 1 if x else 0)
        elif args.regressiontype == "linear":
            data[resp + ".norm"] = data[resp].map(lambda x: 1 if x else -1) * data[resp_conf + ".norm"]
            data_dev[resp + ".norm"] = data_dev[resp].map(lambda x: 1 if x else -1) * data_dev[resp_conf + ".norm"]

    # Shuffle the data
    data = shuffle(data)
    data_dev = shuffle(data_dev)
    data_test = shuffle(data_test)

    # ELMO embeddings
    options_file = args.embeddings + "options/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    weight_file = args.embeddings + "weights/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)
    # elmo = Elmo(options_file, weight_file, 2)

    # pyTorch figures out device to do computation on
    device_to_use = device("cuda:0" if is_available() else "cpu")

    # Prepare all the inputs

    x = [data['Structure'].values.tolist()[i:i + args.batch] for i in range(0, len(data['Structure']), args.batch)]
    x[-1] = x[-1] + x[-2][0:len(x[-2]) - len(x[-1])]

    tokens = [data[token_col].values.tolist()[i:i + args.batch] for i in range(0, len(data[token_col]), args.batch)]
    tokens[-1] = np.append(tokens[-1], tokens[-2][0:len(tokens[-2]) - len(tokens[-1])])

    y = [{attr: (data[attr_map[attr] + ".norm"].values[i:i + args.batch]) for attr in attributes} for i in range(0, len(data[attr_map[attr] + ".norm"].values), args.batch)]
    y[-1] = {attr: np.append(y[-1][attr], y[-2][attr][0:len(y[-2][attr]) - len(y[-1][attr])]) for attr in attributes}

    loss_wts = [{attr: data[attr_conf[attr] + ".norm"].values[i:i + args.batch] for attr in attributes} for i in range(0, len(data[attr_conf[attr]+ ".norm"].values), args.batch)]
    loss_wts[-1] = {attr: np.append(loss_wts[-1][attr], loss_wts[-2][attr][0:len(loss_wts[-2][attr]) - len(loss_wts[-1][attr])]) for attr in attributes}

    # Create dev data

    if args.regressiontype == "linear":
        data_dev_mean = data_dev.groupby('Unique.ID', as_index=False)[dev_cols].mean()
    else:
        data_dev_mean = data_dev.groupby('Unique.ID', as_index=False)[dev_cols].apply(lambda x: dev_mode_group(x, attributes, response, response_conf, attr_map, attr_conf)).reset_index(drop=True)

    data_dev_mean['Structure'] = data_dev_mean['Unique.ID'].map(lambda x: data_dev[data_dev['Unique.ID'] == x]['Structure'].iloc[0])

    dev_x = [data_dev_mean['Structure'].values.tolist()[i:i + args.batch] for i in range(0, len(data_dev_mean['Structure']), args.batch)]
    dev_x[-1] = dev_x[-1] + dev_x[-2][0:len(dev_x[-2]) - len(dev_x[-1])]

    dev_tokens = [data_dev_mean[token_col].values.tolist()[i:i + args.batch] for i in range(0, len(data_dev_mean[token_col]), args.batch)]
    dev_tokens[-1] = np.append(dev_tokens[-1], dev_tokens[-2][0:len(dev_tokens[-2]) - len(dev_tokens[-1])])

    dev_y = {}
    dev_wts = {}
    for attr in attributes:
        dev_y[attr] = [data_dev_mean[attr_map[attr] + ".norm"].values[i:i + args.batch] for i in range(0, len(data_dev_mean[attr_map[attr] + ".norm"].values), args.batch)]
        dev_y[attr][-1] = np.append(dev_y[attr][-1], dev_y[attr][-2][0:len(dev_y[attr][-2]) - len(dev_y[attr][-1])])
        dev_wts[attr] = [data_dev_mean[attr_conf[attr] + ".norm"].values[i:i + args.batch] for i in range(0, len(data_dev_mean[attr_conf[attr] + ".norm"].values), args.batch)]
        dev_wts[attr][-1] = np.append(dev_wts[attr][-1], dev_wts[attr][-2][0:len(dev_wts[attr][-2]) - len(dev_wts[attr][-1])])

    for attr in attributes:
        dev_y[attr] = np.concatenate(dev_y[attr], axis=None)
        dev_wts[attr] = np.concatenate(dev_wts[attr], axis=None)

    # Initialise the model
    trainer = MLPTrainer(embeddings=elmo, device=device_to_use,
                         attributes=attributes, attention=attention_type,
                         regressiontype=args.regressiontype)

    # Training phase
    trainer.fit(X=x, Y=y, loss_wts=loss_wts, tokens=tokens, verbosity=args.verbosity, dev=[dev_x, dev_y, dev_tokens, dev_wts])

    # Save the model
