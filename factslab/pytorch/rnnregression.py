import numpy as np
import pandas as pd
import torch
import torch.autograd
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import LSTM
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss
from scipy.special import huber
from random import shuffle
from collections import Iterable
from factslab.utility import partition
from .childsumtreelstm import *
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from scipy.stats import pearsonr


class RNNRegression(torch.nn.Module):
    """Pytorch module for running the out of an RNN through and MLP

    The most basic use-case for this module is to run an RNN (default:
    LSTM) on some inputs, then predict an output using a regression
    (default: linear) on the final state of that LSTM. More complex
    use-cases are also supported:

    Multi-layered regression (the natural generalization of the
    multi-layer perceptron to arbitrary link functions) can be
    implemented by passing an iterable of ints representing hidden
    state sizes for the regression as `regression_hidden_sizes`.

    End-to-end RNN cascades (using the hidden states of one RNN as the
    inputs to another) can be implemented by passing an iterable of
    RNN pytorch module classes as `rnn_classes`.

    The hidden state sizes, number of layers, and bidirectionality for
    each RNN in a cascade can be specified by passing iterables of
    ints, ints, and bools (respectively) to `rnn_hidden_sizes`,
    `num_rnn_layers`, and `bidirectional` (respectively). Note that,
    if an iterable is passed as any of these three parameters, it must
    have the same length as `rnn_classes`. When an iterable isn't
    passed to any one of these parameters, the same value is used for
    all RNNs in the cascade.

    Parameters
    ----------
    embeddings : numpy.array or NoneType
        a vocab-by-embedding dim matrix (e.g. GloVe)
    embedding_size : int or NoneType
        only specify if not passing a pre-trained embedding
    vocab : list(str) or NoneType
        only specify if not passing a pre-trained embedding
    rnn_classes : subclass RNNBase or iterable(subclass RNNBase)
    rnn_hidden_sizes : int or iterable(int)
        the size of the hidden states in each layer of each
        kind of RNN, going from input (RNN hidden state)
        to output; must be same length as rnn_classes
    num_rnn_layers : int or iterable(int)
        must be same length as rnn_hidden_size
    bidirectional : bool or iterable(bool)
        must be same length as rnn_hidden_size
    attention : bool
        whether to use attention on the final RNN
    regression_hidden_sizes : iterable(int)
        the size of the hidden states in each layer of a
        multilayer regression, going from input (RNN hidden state)
        to output
    device : torch.device
        device(type="cpu") or device(type="cuda:0")
    """

    def __init__(self, embeddings=None, embedding_size=None, vocab=None,
                 rnn_classes=LSTM, rnn_hidden_sizes=300,
                 num_rnn_layers=1, bidirectional=False, attention=False,
                 regression_hidden_sizes=[], output_size=1,
                 device=torch.device(type="cpu"), batch_size=128,
                 attributes=["part"]):
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        # initialize model
        self._initialize_embeddings(embeddings, vocab)
        self._initialize_rnn(rnn_classes, rnn_hidden_sizes,
                             num_rnn_layers, bidirectional)
        self._initialize_regression(attention,
                                    regression_hidden_sizes,
                                    output_size, attributes)

    def _homogenize_parameters(self, rnn_classes, rnn_hidden_sizes,
                               num_rnn_layers, bidirectional):

        iterables = [p for p in [rnn_classes, rnn_hidden_sizes,
                                 num_rnn_layers, bidirectional]
                     if isinstance(p, Iterable)]
        max_length = max([len(p) for p in iterables]) if iterables else 1

        if not isinstance(rnn_classes, Iterable):
            self.rnn_classes = [rnn_classes] * max_length
        else:
            self.rnn_classes = rnn_classes

        if not isinstance(rnn_hidden_sizes, Iterable):
            self.rnn_hidden_sizes = [rnn_hidden_sizes] * max_length
        else:
            self.rnn_hidden_sizes = rnn_hidden_sizes

        if not isinstance(num_rnn_layers, Iterable):
            self.num_rnn_layers = [num_rnn_layers] * max_length
        else:
            self.num_rnn_layers = num_rnn_layers

        if not isinstance(bidirectional, Iterable):
            self.bidirectional = [bidirectional] * max_length
        else:
            self.bidirectional = bidirectional

    def _validate_parameters(self):
        try:
            assert len(self.rnn_classes) == len(self.rnn_hidden_sizes)
            assert len(self.rnn_classes) == len(self.num_rnn_layers)
            assert len(self.rnn_classes) == len(self.bidirectional)
        except AssertionError:
            msg = "rnn_classes, rnn_hidden_sizes," +\
                  "num_rnn_layers, and bidirectional" +\
                  "must be non-iterable or the same length"
            raise ValueError(msg)

    def _initialize_embeddings(self, embeddings, vocab):
        # set embedding hyperparameters
        if embeddings is None:
            self.vocab = vocab
            self.num_embeddings = len(self.vocab)
            self.embedding_size = embedding_size
        else:
            self.num_embeddings, self.embedding_size = embeddings.shape
            self.vocab = embeddings.index

        # define embedding layer
        self.embeddings = torch.nn.Embedding(self.num_embeddings,
                                             self.embedding_size,
                                             max_norm=None,
                                             norm_type=2,
                                             scale_grad_by_freq=False,
                                             sparse=False)

        # copy the embeddings into the embedding layer
        if embeddings is not None:
            embeddings_torch = torch.from_numpy(embeddings.values)
            self.embeddings.weight.data.copy_(embeddings_torch)
        # Turn off gradients on embeddings
        self.embeddings.weight.requires_grad = False
        # construct the hash
        self.vocab_hash = {w: i for i, w in enumerate(self.vocab)}

    def _initialize_rnn(self, rnn_classes, rnn_hidden_sizes,
                        num_rnn_layers, bidirectional):

        self._homogenize_parameters(rnn_classes, rnn_hidden_sizes,
                                    num_rnn_layers, bidirectional)
        self._validate_parameters()

        output_size = self.embedding_size
        self.rnns = []

        params_zipped = zip(self.rnn_classes, self.rnn_hidden_sizes,
                            self.num_rnn_layers, self.bidirectional)

        for i, (rnn_class, hsize, lnum, bi) in enumerate(params_zipped):
            input_size = output_size
            rnn = rnn_class(input_size=input_size,
                            hidden_size=hsize,
                            num_layers=lnum,
                            bidirectional=bi,
                            batch_first=True)
            rnn = rnn.to(self.device)
            self.rnns.append(rnn)
            varname = '_rnn_' + str(i)
            RNNRegression.__setattr__(self, varname, rnn)
            output_size = hsize * 2 if bi else hsize

        self.rnn_output_size = output_size

    def _initialize_regression(self, attention, hidden_sizes, output_size, attributes):
        self.attributes = attributes
        self.linear_maps = {}

        self.attention = attention
        self.dropout = Dropout()
        last_size = self.rnn_output_size
        if self.attention:
            if isinstance(self.rnns[0], LSTM):
                self.attention_map = Parameter(torch.zeros(self.batch_size,
                                                           last_size))
            else:
                self.attention_map = Parameter(torch.zeros(last_size))

        for attr in self.attributes:
            self.linear_maps[attr] = []
            for i, h in enumerate(hidden_sizes):
                linmap = torch.nn.Linear(last_size, h)
                linmap = linmap.to(self.device)
                self.linear_maps[attr].append(linmap)
                varname = '_linear_map' + attr + str(i)
                RNNRegression.__setattr__(self, varname, linmap)
                last_size = h

            linmap = torch.nn.Linear(last_size, output_size)
            linmap = linmap.to(self.device)
            self.linear_maps[attr].append(linmap)
            varname = '_linear_map' + attr + str(len(hidden_sizes))
            RNNRegression.__setattr__(self, varname, linmap)

    def forward(self, structures, tokens, lengths, mode):
        """
        Parameters
        ----------
        structures : iterable(object)
           the structures to be used in determining the RNNs
           composition path. Each element must correspond to the
           corresponding RNN in a cascade, or in the case of a trivial
           cascade (a single RNN followed by regression), `structures`
           must be a singleton iterable. When the relevant RNN in a
           cascade is a linear-chain RNN, the structure in the
           corresponding position of this parameter is ignored
        targets: list
            A list of all the targets in the batch. This will be modified only
            if the rnn_class is LSTM(since the order will be modified
            during padding). Otherwise it is returned as is.
        """

        try:
            words = structures.words()
        except AttributeError:
            # assert all([isinstance(w, str) for w in structures])
            words = structures
        except AssertionError:
            msg = "first structure in sequence must either" +\
                  "implement a words() method or itself be" +\
                  "a sequence of words"
            raise ValueError(msg)

        if isinstance(words[0], list):
            self.has_batch_dim = True
        else:
            self.has_batch_dim = False

        inputs = self._get_inputs(words)
        inputs = self._preprocess_inputs(inputs)

        h_all, h_last = self._run_rnns(inputs, structures, lengths)

        if self.attention:
            h_last = self._run_attention(h_all)
        else:
            h_last = self.choose_timestep(h_all, tokens)

        h_last = self._run_regression(h_last, mode)

        y_hat = self._postprocess_outputs(h_last)

        return y_hat

    def _run_rnns(self, inputs, structures, lengths):
        '''
            Run desired rnns
        '''
        for rnn, structure in zip(self.rnns, [structures]):
            if isinstance(rnn, ChildSumTreeLSTM):
                h_all, h_last = rnn(inputs, structure)
            elif isinstance(rnn, LSTM) and lengths is not None:
                packed = pack_padded_sequence(inputs, lengths, batch_first=True)
                h_all, (h_last, c_last) = rnn(packed)
                h_all, _ = pad_packed_sequence(h_all, batch_first=True)
            else:
                h_all, (h_last, c_last) = rnn(inputs)
            inputs = h_all.squeeze()

        return h_all, h_last

    def _run_attention(self, h_all, return_weights=False):
        if not self.has_batch_dim:
            att_raw = torch.mm(h_all, self.attention_map[:, None])
            att = F.softmax(att_raw.squeeze(), dim=0)

            if return_weights:
                return att
            else:
                return torch.mm(att[None, :], h_all).squeeze()
        else:
            att_raw = torch.bmm(h_all, self.attention_map[:, :, None])
            att = F.softmax(att_raw.squeeze(), dim=0)

            if return_weights:
                return att
            else:
                return torch.bmm(att[:, None, :], h_all).squeeze()

    def _run_regression(self, h_in, mode):
        # Neural davidsonian(simple)
        # h_shared = F.relu(torch.mm(self.attr_shared, h_last.unsqueeze(1)))
        # h = {attr: None for attr in self.attributes}
        # for attr in self.attributes:
        #     h[attr] = torch.mm(torch.transpose(h_shared, 0, 1), self.attr_sp[attr]).squeeze()
        h_last = {}
        # if mode == "train":
        #     h_in = self.dropout(h_in)
        for attr in self.attributes:
            h_last[attr] = h_in
            for i, linear_map in enumerate(self.linear_maps[attr]):
                if i:
                    h_last[attr] = self._regression_nonlinearity(h_last[attr])
                h_last[attr] = linear_map(h_last[attr])
        return h_last

    def _regression_nonlinearity(self, x):
        return F.relu(x)

    def _preprocess_inputs(self, inputs):
        """Apply some function(s) to the input embeddings

        This is included to allow for an easy preprocessing hook for
        RNNRegression subclasses. For instance, we might want to
        apply a tanh to the inputs to make them look more like features
        """
        return inputs

    def _postprocess_outputs(self, outputs):
        """Apply some function(s) to the output value(s)"""
        for attr in self.attributes:
            outputs[attr] = outputs[attr].squeeze()
        return outputs

    def choose_timestep(self, output, idxs):
        # Index extraction for each sequence
        idx = (idxs - 1).view(-1, 1).expand(output.size(0), output.size(2)).unsqueeze(1).to(self.device)
        return output.gather(1, idx).squeeze()

    def _get_inputs(self, inputs):
        if self.has_batch_dim:
            indices = []
            for sent in inputs:
                indices.append([self.vocab_hash[word] for word in sent])
        else:
            indices = [self.vocab_hash[word] for word in inputs]
        indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.embeddings(indices)

    def word_embeddings(self, words=[]):
        """Extract the tuned word embeddings

        If an empty list is passed, all word embeddings are returned

        Parameters
        ----------
        words : list(str)
            The words to get the embeddings for

        Returns
        -------
        pandas.DataFrame
        """
        words = words if words else self.vocab
        embeddings = self._get_inputs(words).data.cpu().numpy()

        return pd.DataFrame(embeddings, index=words)

    def attention_weights(self, structures):
        """Compute what the LSTM regression is attending to

        The weights that are returned are only for the structures used
        in the last LSTM. This is because that is the only place that
        attention is implemented - i.e. right befoe passing the LSTM
        outputs to a regression layer.

        Parameters
        ----------
        structures : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN

        Returns
        -------
        pytorch.Tensor
        """
        try:
            assert self.attention
        except AttributeError:
            raise AttributeError('attention not used')

        try:
            words = structures[0].words()
        except AttributeError:
            assert all([isinstance(w, str)
                        for w in structures[0]])
            words = structures[0]
        except AssertionError:
            msg = "first structure in sequence must either" +\
                  "implement a words() method or itself be" +\
                  "a sequence of words"
            raise ValueError(msg)

        inputs = self._get_inputs(words)
        inputs = self._preprocess_inputs(inputs)

        h_all, h_last = self._run_rnns(inputs, structures)

        return self._run_attention(h_all, return_weights=True)


class RNNRegressionTrainer(object):

    loss_function_map = {"linear": MSELoss,
                         "robust": L1Loss,
                         "robust_smooth": SmoothL1Loss,
                         "multinomial": CrossEntropyLoss}

    def __init__(self, regression_type="linear",
                 optimizer_class=torch.optim.Adam,
                 device=torch.device(type="cpu"), epochs=10,
                 rnn_classes=LSTM, attributes=["acceptability"], **kwargs):
        self._regression_type = regression_type
        self._optimizer_class = optimizer_class
        self.epochs = epochs
        self.attributes = attributes
        self._init_kwargs = kwargs
        self.rnn_classes = rnn_classes
        self._continuous = regression_type != "multinomial"
        self.device = device

    def _initialize_trainer_regression(self):
        if self._continuous:
            self._regression = RNNRegression(device=self.device,
                                             rnn_classes=self.rnn_classes,
                                             attributes=self.attributes,
                                             **self._init_kwargs)
        else:
            if self.rnn_classes =='LSTM':
                output_size = np.unique(self._Y[0]).shape[0]
            else:
                output_size = np.unique(self._Y).shape[0]
            self._regression = RNNRegression(output_size=output_size,
                                             device=self.device,
                                             rnn_classes=self.rnn_classes,
                                             attributes=self.attributes,
                                             **self._init_kwargs)

        lf_class = self.__class__.loss_function_map[self._regression_type]
        self._loss_function = lf_class()

        self._regression = self._regression.to(self.device)
        self._loss_function = self._loss_function.to(self.device)

    def fit(self, X, Y, lengths, dev, batch_size=100, verbosity=1, **kwargs):
        """Fit the LSTM regression

        Parameters
        ----------
        X : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN
        Y : numpy.array(Number)
            a matrix of dependent variables
        batch_size : int (default: 100)
        verbosity : int (default: 1)
            how often to print metrics (never if 0)
        """

        self._X, self._Y = X, Y
        dev_x, dev_y, dev_lengths = dev
        self._initialize_trainer_regression()

        for name, param in self._regression.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        parameters = [p for p in self._regression.parameters() if p.requires_grad]
        optimizer = self._optimizer_class(parameters, **kwargs)
        self._Y_logprob = {}
        if not self._continuous:
            for attr in self.attributes:
                Y_counts = np.bincount([y for batch in self._Y[attr] for y in batch])
                self._Y_logprob[attr] = np.log(Y_counts) - np.log(np.sum(Y_counts))

        # each element is of the form ((struct1, struct2, ...),
        #                              target)
        structures_targets = list(zip(self._X, lengths))
        loss_trace = []
        targ_trace = {}
        pred_trace = {}
        early_stop = {}
        for attr in self.attributes:
            targ_trace[attr] = np.array([])
            pred_trace[attr] = np.array([])
            early_stop = [0.0]
        epoch = 0
        while epoch < self.epochs:
            epoch += 1
            print("Epoch:", epoch, "\n")
            print("Progress" + "\t Metrics")
            losses = []

            total = len(self._Y['acceptability'])
            for i, structs_targs_batch in enumerate(structures_targets):
                optimizer.zero_grad()
                if self.rnn_classes == LSTM:
                    structs, lengths = structs_targs_batch

                    lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
                    targs = {}
                    losses = {}
                    for attr in self.attributes:
                        targs[attr] = self._Y[attr][i]
                        targ_trace[attr] = np.append(targ_trace[attr], targs[attr])
                        if self._continuous:
                            targs[attr] = torch.tensor(targs[attr], dtype=torch.float, device=self.device)
                        else:
                            targs[attr] = torch.tensor(targs[attr], dtype=torch.long, device=self.device)

                    predicted = self._regression(structures=structs,
                                                 tokens=lengths,
                                                 lengths=lengths,
                                                 mode="train")
                    for attr in self.attributes:
                        losses[attr] = self._loss_function(predicted[attr], targs[attr])
                        pred_trace[attr] = np.append(pred_trace[attr], predicted[attr].detach())

                else:
                    structs_targs = list(zip(structs_targs_batch[0],
                                             structs_targs_batch[1]))
                    for struct, targ in structs_targs:
                        targ_trace.append(targ)

                        if self._continuous:
                            targ = torch.tensor([targ], dtype=torch.float)
                        else:
                            targ = torch.tensor([int(targ)], dtype=torch.long)

                        targ = targ.to(self.device)
                        predicted, targ = self._regression(struct, targ)
                        if self._continuous:
                            loss = self._loss_function(predicted, targ)
                        else:
                            loss = self._loss_function(predicted[None, :], targ)

                        losses.append(loss)

                loss = sum(losses.values()) / len(losses)
                loss_trace.append(loss.item())

                loss.backward()
                optimizer.step()

                # TODO: generalize for non-linear regression
                if verbosity:
                    if not (i + 1) % verbosity:
                        progress = "{:.0f}".format(((i) / total) * 100)
                        self._print_metric(progress, loss_trace, targ_trace, pred_trace)
                        loss_trace = []
                        targ_trace = {}
                        pred_trace = {}
                        for attr in self.attributes:
                            targ_trace[attr] = np.array([])
                            pred_trace[attr] = np.array([])


            # Implement early stopping here
            print("VALIDATION")

            predictions = self.predict(X=dev_x, dev_lengths=dev_lengths)

            for attr in self.attributes:
                outputs = predictions[attr]
                targets = [y for batch in dev_y[attr] for y in batch]
                correlation = pearsonr(outputs, targets)[0]
            corrs = []
            for attr in self.attributes:
                print(attr)
                print("Correlation:", correlation)
                corrs.append(correlation)
            early_stop.append(np.mean(corrs))
            print("Difference in mean corr:", early_stop[-1] - early_stop[-2])
            if (early_stop[-1] - early_stop[-2]) < 0:
                break

    def _print_metric(self, progress, loss_trace, targ_trace, pred_trace):

        sigdig = 3
        Y_flat = [y for batch in self._Y['acceptability'] for y in batch]
        if self._continuous:
            resid_mean = np.mean(loss_trace)

            if self._regression_type == "linear":
                targ_var = np.mean(np.square(np.array(targ_trace['acceptability']) - np.mean(Y_flat)))
                r2 = 1. - (resid_mean / targ_var)
                corr = pearsonr(targ_trace['acceptability'], pred_trace['acceptability'])[0]
                print(progress + "%" + '\t\t residual variance:\t', np.round(resid_mean, sigdig), '\n',
                      ' \t\t total variance:\t', np.round(targ_var, sigdig), '\n',
                      ' \t\t r-squared:\t\t', np.round(r2, sigdig), '\n',
                      ' \t\t correlation:\t\t', np.round(corr, sigdig), '\n')
                    # ' \t\t total variance:\t', np.round(targ_var, sigdig), '\n'
            elif self._regression_type == "robust":
                ae = np.abs(targ_trace - np.median(Y_flat))
                mae = np.mean(ae)
                pmae = 1. - (resid_mean / mae)

                print(progress + "%" + '\t\t residual absolute error:\t', np.round(resid_mean, sigdig), '\n',
                      ' \t\t total absolute error:\t\t', np.round(mae, sigdig), '\n',
                      ' \t\t proportion absolute error:\t', np.round(pmae, sigdig), '\n')

            elif self._regression_type == "robust_smooth":
                ae = huber(1., targ_trace - np.median(Y_flat))
                mae = np.mean(ae)
                pmae = 1. - (resid_mean / mae)

                print(progress + "%" + '\t\t residual absolute error:\t', np.round(resid_mean, sigdig), '\n',
                      ' \t\t total absolute error:\t\t', np.round(mae, sigdig), '\n',
                      ' \t\t proportion absolute error:\t', np.round(pmae, sigdig), '\n')

        else:
            model_mean_neglogprob = np.mean(loss_trace)
            targ_mean_neglogprob = -np.mean([self._Y_logprob[x] for x in targ_trace])
            pnlp = 1. - (model_mean_neglogprob / targ_mean_neglogprob)

            print(progress + "%" + '\t\t residual mean cross entropy:\t', np.round(model_mean_neglogprob, sigdig), '\n',
                  ' \t\t total mean cross entropy:\t', np.round(targ_mean_neglogprob, sigdig), '\n',
                  ' \t\t proportion entropy explained:\t', np.round(pnlp, sigdig), '\n')

    def predict(self, X, dev_lengths):
        """Predict using the LSTM regression

        Parameters
        ----------
        X : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN
        """

        predictions = {'acceptability': []}
        for struct, lengths in zip(X, dev_lengths):
            lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
            predictions['acceptability'] += self._regression(struct, tokens=lengths, lengths=lengths, mode="dev")['acceptability'].tolist()
        return predictions
        # if self._continuous:
        #     return np.array([p.data.cpu().numpy() for p in predictions])
        # else:
        #     dist = np.array([p.data.cpu().numpy() for p in predictions])
        #     return np.where(dist == np.max(dist, axis=1)[:, None])

    def attention_weights(self, X):
        """Compute what the LSTM regression is attending to

        Parameters
        ----------
        X : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN

        Returns
        -------
        list(np.array)
        """
        attention = [self._regression.attention_weights(struct)
                     for struct in zip(*X)]
        return [a.data.cpu().numpy() for a in attention]

    def word_embeddings(self, words=[]):
        """Extract the tuned word embeddings

        If an empty list is passed, all word embeddings are returned

        Parameters
        ----------
        words : list(str)
            The words to get the embeddings for

        Returns
        -------
        pandas.DataFrame
        """
        return self._regression.word_embeddings(words)
