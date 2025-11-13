"""
Calibration Plotting Module for Multiclass Classification
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from calibration import get_calibration_data, compute_ece, compute_mce


def plot_calibration(probs_before, probs_after, labels, output_path, n_bins=15):
    """
    Plot calibration (reliability) diagram before and after temperature scaling.

    Args:
        probs_before: Predicted probabilities before calibration (N, C)
        probs_after: Predicted probabilities after calibration (N, C)
        labels: True labels (N,)
        output_path: Path to save the plot
        n_bins: Number of bins for calibration

    Returns:
        None (saves plot to file)
    """
    # Get calibration data
    bin_acc_before, bin_conf_before, bin_counts_before = get_calibration_data(
        probs_before, labels, n_bins
    )
    bin_acc_after, bin_conf_after, bin_counts_after = get_calibration_data(
        probs_after, labels, n_bins
    )

    # Compute ECE and MCE
    ece_before = compute_ece(probs_before, labels, n_bins)
    mce_before = compute_mce(probs_before, labels, n_bins)
    ece_after = compute_ece(probs_after, labels, n_bins)
    mce_after = compute_mce(probs_after, labels, n_bins)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot before calibration
    ax1.bar(bin_conf_before, bin_acc_before, width=1.0/n_bins,
            alpha=0.7, edgecolor='black', label='Outputs')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Before Calibration\nECE={ece_before:.4f}, MCE={mce_before:.4f}',
                  fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Plot after calibration
    ax2.bar(bin_conf_after, bin_acc_after, width=1.0/n_bins,
            alpha=0.7, edgecolor='black', color='green', label='Calibrated outputs')
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'After Temperature Scaling\nECE={ece_after:.4f}, MCE={mce_after:.4f}',
                  fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Calibration plot saved to: {output_path}")
    print(f"Before calibration - ECE: {ece_before:.4f}, MCE: {mce_before:.4f}")
    print(f"After calibration - ECE: {ece_after:.4f}, MCE: {mce_after:.4f}")

    return {
        'ece_before': ece_before,
        'mce_before': mce_before,
        'ece_after': ece_after,
        'mce_after': mce_after
    }


def save_calibration_metrics(metrics_dict, output_path):
    """
    Save calibration metrics to a text file.

    Args:
        metrics_dict: Dictionary with calibration metrics
        output_path: Path to save the metrics file

    Returns:
        None
    """
    with open(output_path, 'w') as f:
        f.write("Calibration Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write("Before Temperature Scaling:\n")
        f.write(f"  Expected Calibration Error (ECE): {metrics_dict['ece_before']:.6f}\n")
        f.write(f"  Maximum Calibration Error (MCE): {metrics_dict['mce_before']:.6f}\n\n")
        f.write("After Temperature Scaling:\n")
        f.write(f"  Expected Calibration Error (ECE): {metrics_dict['ece_after']:.6f}\n")
        f.write(f"  Maximum Calibration Error (MCE): {metrics_dict['mce_after']:.6f}\n\n")
        f.write(f"Temperature: {metrics_dict.get('temperature', 'N/A')}\n")

    print(f"Calibration metrics saved to: {output_path}")
