"""
Multiclass GCN Model with Calibration Support
"""
import math
import time
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from sklearn import metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_onehot_multiclass(labels):
    """
    Encode labels to one-hot format for multiclass classification.

    Args:
        labels: Array of label strings

    Returns:
        labels_onehot: One-hot encoded labels
        classes_dict: Dictionary mapping class names to one-hot vectors
    """
    # Get unique classes sorted
    classes = sorted(list(set(labels)))
    # Filter out 'Unknown' if present
    classes = [c for c in classes if c != 'Unknown']

    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)

    return labels_onehot, classes_dict


def normalize(mx):
    """Normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphConvolution(Module):
    """Graph Convolution Layer."""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + \
               ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    """Graph Convolutional Network for multiclass classification."""

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = torch.nn.functional.relu(self.gc1(x, adj))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return torch.nn.functional.log_softmax(x, dim=1)

    def forward_logits(self, x, adj):
        """Forward pass returning logits (before log_softmax) for calibration."""
        x = torch.nn.functional.relu(self.gc1(x, adj))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x  # Return logits without log_softmax


def accuracy(output, labels):
    """Calculate accuracy for multiclass classification."""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def multiclass_metrics(output, labels, n_classes):
    """
    Calculate multiclass classification metrics.

    Args:
        output: Model output (log probabilities)
        labels: True labels
        n_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    output = torch.exp(output).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    preds = np.argmax(output, axis=1)
    acc = np.mean(preds == labels_np)

    # Calculate per-class metrics if more than 2 classes
    if n_classes > 2:
        # Macro-averaged metrics
        try:
            f1_macro = metrics.f1_score(labels_np, preds, average='macro', zero_division=0)
            precision_macro = metrics.precision_score(labels_np, preds, average='macro', zero_division=0)
            recall_macro = metrics.recall_score(labels_np, preds, average='macro', zero_division=0)
        except:
            f1_macro = 0.0
            precision_macro = 0.0
            recall_macro = 0.0

        return {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        }
    else:
        # Binary classification - compute AUC
        try:
            if len(np.unique(labels_np)) > 1:
                auc = metrics.roc_auc_score(labels_np, output[:, 1])
            else:
                auc = 0.0
        except:
            auc = 0.0

        return {
            'accuracy': acc,
            'auc': auc
        }


def train_epoch(model, optimizer, features, adj, labels, train_idx):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = torch.nn.functional.nll_loss(output[train_idx], labels[train_idx])
    loss_train.backward()
    optimizer.step()
    return loss_train.item()


def evaluate(model, features, adj, labels, eval_idx):
    """Evaluate model on given indices."""
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss = torch.nn.functional.nll_loss(output[eval_idx], labels[eval_idx])
        acc = accuracy(output[eval_idx], labels[eval_idx])

    return loss.item(), acc.item(), output[eval_idx]


def get_predictions(model, features, adj, indices):
    """Get model predictions (probabilities) for given indices."""
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        probs = torch.exp(output[indices])

    return probs


def get_logits(model, features, adj, indices):
    """Get model logits (before softmax) for calibration."""
    model.eval()
    with torch.no_grad():
        logits = model.forward_logits(features, adj)

    return logits[indices]
