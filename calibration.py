"""
Temperature Scaling Calibration Module for Multiclass Classification
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import log_loss


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.

    Reference:
    Guo et al. (2017). On Calibration of Modern Neural Networks.
    """

    def __init__(self, n_classes):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.n_classes = n_classes

    def forward(self, logits):
        """
        Apply temperature scaling to logits.

        Args:
            logits: Model output before softmax (N, C)

        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature

    def fit(self, logits, labels, max_iter=50, lr=0.01):
        """
        Fit temperature parameter using validation set.

        Args:
            logits: Model output logits (N, C) - torch tensor or numpy array
            labels: True labels (N,) - torch tensor or numpy array
            max_iter: Maximum optimization iterations
            lr: Learning rate

        Returns:
            Optimized temperature value
        """
        # Convert to torch tensors if needed
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()

        # Optimization
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        return self.temperature.item()

    def predict_proba(self, logits):
        """
        Get calibrated probabilities.

        Args:
            logits: Model output logits (N, C)

        Returns:
            Calibrated probabilities (N, C)
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()

        with torch.no_grad():
            calibrated_logits = self.forward(logits)
            probs = F.softmax(calibrated_logits, dim=1)

        return probs.numpy() if isinstance(logits, torch.Tensor) else probs


def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE) for multiclass classification.

    Args:
        probs: Predicted probabilities (N, C)
        labels: True labels (N,)
        n_bins: Number of bins for calibration

    Returns:
        ECE score
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    n_samples, n_classes = probs.shape

    # Get predicted class and confidence
    pred_classes = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (pred_classes == labels).astype(float)

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            bin_size = np.sum(bin_mask)
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return ece


def compute_mce(probs, labels, n_bins=15):
    """
    Compute Maximum Calibration Error (MCE) for multiclass classification.

    Args:
        probs: Predicted probabilities (N, C)
        labels: True labels (N,)
        n_bins: Number of bins for calibration

    Returns:
        MCE score
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    n_samples, n_classes = probs.shape

    # Get predicted class and confidence
    pred_classes = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (pred_classes == labels).astype(float)

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mce = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ce = np.abs(bin_acc - bin_conf)
            mce = max(mce, ce)

    return mce


def get_calibration_data(probs, labels, n_bins=15):
    """
    Get calibration data for plotting reliability diagram.

    Args:
        probs: Predicted probabilities (N, C)
        labels: True labels (N,)
        n_bins: Number of bins

    Returns:
        bin_accuracies, bin_confidences, bin_counts
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Get predicted class and confidence
    pred_classes = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (pred_classes == labels).astype(float)

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
        else:
            bin_acc = 0.0
            bin_conf = (bins[i] + bins[i+1]) / 2

        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)
        bin_counts.append(bin_count)

    return np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)
