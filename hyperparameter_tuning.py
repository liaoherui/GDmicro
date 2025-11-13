"""
Hyperparameter Tuning Module with Nested Cross-Validation and Random Search
"""
import numpy as np
import torch
import random
from sklearn.model_selection import StratifiedKFold
from gcn_model_train_mode import GCN, encode_onehot
import scipy.sparse as sp


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_hyperparameter_space():
    """
    Define hyperparameter search space for GCN model.

    Returns:
        Dictionary of hyperparameter ranges
    """
    return {
        'hidden_dim': [16, 32, 64, 128],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3],
        'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
        'epochs': [100, 150, 200]
    }


def sample_hyperparameters(hp_space, n_samples=1):
    """
    Sample random hyperparameter configurations.

    Args:
        hp_space: Dictionary of hyperparameter ranges
        n_samples: Number of configurations to sample

    Returns:
        List of hyperparameter dictionaries
    """
    configs = []
    for _ in range(n_samples):
        config = {}
        for param, values in hp_space.items():
            config[param] = random.choice(values)
        configs.append(config)
    return configs


def train_with_config(features, adj, labels, train_idx, val_idx, config, seed=42):
    """
    Train GCN model with specific hyperparameter configuration.

    Args:
        features: Node features
        adj: Adjacency matrix
        labels: Node labels
        train_idx: Training indices
        val_idx: Validation indices
        config: Hyperparameter configuration dictionary
        seed: Random seed

    Returns:
        Validation accuracy
    """
    setup_seed(seed)

    # Create model with config
    model = GCN(
        nfeat=features.shape[1],
        nhid=config['hidden_dim'],
        nclass=labels.max().item() + 1,
        dropout=config['dropout']
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Training loop
    best_val_acc = 0.0
    for epoch in range(config['epochs']):
        # Train
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = torch.nn.functional.nll_loss(output[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            pred = output[val_idx].max(1)[1]
            val_acc = pred.eq(labels[val_idx]).sum().item() / len(val_idx)
            if val_acc > best_val_acc:
                best_val_acc = val_acc

    return best_val_acc


def nested_cv_random_search(features, adj, labels, train_indices, cohort_labels,
                             n_inner_folds=3, n_iter=10, seed=42):
    """
    Perform hyperparameter optimization using nested cross-validation with random search.

    Args:
        features: Node features
        adj: Adjacency matrix
        labels: Node labels
        train_indices: Indices for the outer fold training set
        cohort_labels: Cohort labels for each sample (for inner CV splits)
        n_inner_folds: Number of folds for inner CV
        n_iter: Number of random search iterations
        seed: Random seed

    Returns:
        Best hyperparameter configuration and its average validation accuracy
    """
    setup_seed(seed)

    # Get hyperparameter space
    hp_space = get_hyperparameter_space()

    # Sample random configurations
    configs = sample_hyperparameters(hp_space, n_samples=n_iter)

    # Get cohorts for inner CV
    train_cohorts = cohort_labels[train_indices]
    unique_cohorts = np.unique(train_cohorts)

    # If we have enough cohorts, use leave-one-cohort-out for inner CV
    # Otherwise use stratified k-fold
    if len(unique_cohorts) >= n_inner_folds:
        use_cohort_cv = True
    else:
        use_cohort_cv = False
        inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)

    best_config = None
    best_avg_acc = 0.0
    config_scores = []

    print(f"\nPerforming random search with {n_iter} iterations...")

    for i, config in enumerate(configs):
        fold_accs = []

        if use_cohort_cv:
            # Leave-one-cohort-out for inner CV
            for test_cohort in unique_cohorts[:n_inner_folds]:
                inner_train_mask = train_cohorts != test_cohort
                inner_val_mask = train_cohorts == test_cohort

                inner_train_idx = train_indices[inner_train_mask]
                inner_val_idx = train_indices[inner_val_mask]

                if len(inner_val_idx) == 0:
                    continue

                acc = train_with_config(features, adj, labels,
                                        inner_train_idx, inner_val_idx,
                                        config, seed)
                fold_accs.append(acc)
        else:
            # Stratified k-fold for inner CV
            train_labels = labels[train_indices].numpy()
            for inner_train_idx_rel, inner_val_idx_rel in inner_cv.split(
                    np.zeros(len(train_indices)), train_labels):
                inner_train_idx = train_indices[inner_train_idx_rel]
                inner_val_idx = train_indices[inner_val_idx_rel]

                acc = train_with_config(features, adj, labels,
                                        inner_train_idx, inner_val_idx,
                                        config, seed)
                fold_accs.append(acc)

        avg_acc = np.mean(fold_accs)
        config_scores.append((config, avg_acc))

        print(f"Config {i+1}/{n_iter}: Avg Accuracy = {avg_acc:.4f}")
        print(f"  Params: {config}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_config = config

    print(f"\nBest configuration found:")
    print(f"  Average Accuracy: {best_avg_acc:.4f}")
    print(f"  Params: {best_config}")

    return best_config, best_avg_acc, config_scores
