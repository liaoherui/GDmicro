"""
Leave-One-Cohort-Out Cross-Validation Module
"""
import numpy as np
import pandas as pd


class LeaveOneCohortOut:
    """
    Leave-One-Cohort-Out cross-validator.

    This cross-validation strategy leaves out samples from one cohort/study
    as the test set, and uses all other cohorts for training.
    """

    def __init__(self):
        self.n_splits = None
        self.cohorts = None

    def get_n_splits(self, X=None, y=None, cohorts=None):
        """
        Get the number of splits.

        Args:
            X: Features (not used, for sklearn compatibility)
            y: Labels (not used, for sklearn compatibility)
            cohorts: Cohort labels for each sample

        Returns:
            Number of splits
        """
        if cohorts is None:
            raise ValueError("cohorts parameter is required")

        self.cohorts = np.array(cohorts)
        unique_cohorts = np.unique(self.cohorts)
        # Filter out 'Unknown' cohorts if present
        unique_cohorts = [c for c in unique_cohorts if c != 'Unknown']
        self.n_splits = len(unique_cohorts)
        return self.n_splits

    def split(self, X, y=None, cohorts=None):
        """
        Generate indices to split data into training and test set.

        Args:
            X: Features array
            y: Labels array (optional)
            cohorts: Cohort labels for each sample

        Yields:
            train_idx, test_idx: Indices for train and test sets
        """
        if cohorts is None:
            raise ValueError("cohorts parameter is required")

        self.cohorts = np.array(cohorts)
        unique_cohorts = np.unique(self.cohorts)
        # Filter out 'Unknown' cohorts if present
        unique_cohorts = [c for c in unique_cohorts if c != 'Unknown']
        unique_cohorts = sorted(unique_cohorts)

        for test_cohort in unique_cohorts:
            # Create masks for train and test
            test_mask = self.cohorts == test_cohort
            train_mask = ~test_mask

            # Get indices
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(train_mask)[0]

            # Skip if test set is empty
            if len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_cohort_info(self, cohorts):
        """
        Get information about cohorts.

        Args:
            cohorts: Cohort labels

        Returns:
            Dictionary with cohort information
        """
        cohorts = np.array(cohorts)
        unique_cohorts = np.unique(cohorts)
        unique_cohorts = [c for c in unique_cohorts if c != 'Unknown']

        cohort_info = {}
        for cohort in unique_cohorts:
            cohort_mask = cohorts == cohort
            cohort_info[cohort] = {
                'count': np.sum(cohort_mask),
                'indices': np.where(cohort_mask)[0]
            }

        return cohort_info


def extract_cohorts_from_meta(meta_file):
    """
    Extract cohort information from metadata file.

    Args:
        meta_file: Path to metadata file

    Returns:
        Array of cohort labels
    """
    cohorts = []
    with open(meta_file, 'r') as f:
        # Skip header
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                break
            # Metadata format: sample_id class disease study
            parts = line.split()
            if len(parts) >= 4:
                cohort = parts[3]  # study column
            else:
                cohort = 'Unknown'
            cohorts.append(cohort)

    return np.array(cohorts)


def create_cohort_splits_for_data(features, labels, cohorts):
    """
    Create leave-one-cohort-out splits for given data.

    Args:
        features: Feature array
        labels: Label array
        cohorts: Cohort labels

    Returns:
        List of (train_idx, test_idx, test_cohort_name) tuples
    """
    loco = LeaveOneCohortOut()
    splits = []

    unique_cohorts = np.unique(cohorts)
    unique_cohorts = [c for c in unique_cohorts if c != 'Unknown']

    for train_idx, test_idx in loco.split(features, labels, cohorts):
        test_cohort = cohorts[test_idx[0]]  # Get cohort name from first test sample
        splits.append((train_idx, test_idx, test_cohort))

    return splits


def split_validation_from_training(train_idx, labels, cohorts, val_ratio=0.2, seed=42):
    """
    Split validation set from training set for calibration.

    For cohort-based CV, we take a portion of the training data for validation
    (used for temperature scaling calibration).

    Args:
        train_idx: Training indices
        labels: All labels
        cohorts: All cohort labels
        val_ratio: Ratio of training data to use for validation
        seed: Random seed

    Returns:
        new_train_idx, val_idx
    """
    np.random.seed(seed)

    train_cohorts = cohorts[train_idx]
    train_labels = labels[train_idx]

    unique_train_cohorts = np.unique(train_cohorts)
    unique_train_cohorts = [c for c in unique_train_cohorts if c != 'Unknown']

    # If we have multiple cohorts, take one cohort for validation
    # Otherwise, do random split stratified by label
    if len(unique_train_cohorts) > 1:
        # Take one cohort as validation
        val_cohort = np.random.choice(unique_train_cohorts)
        val_mask = train_cohorts == val_cohort
        new_train_mask = ~val_mask

        val_idx = train_idx[val_mask]
        new_train_idx = train_idx[new_train_mask]
    else:
        # Random stratified split
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(train_idx))
        new_train_indices, val_indices = train_test_split(
            indices,
            test_size=val_ratio,
            stratify=train_labels,
            random_state=seed
        )
        new_train_idx = train_idx[new_train_indices]
        val_idx = train_idx[val_indices]

    return new_train_idx, val_idx
