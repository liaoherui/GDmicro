"""
Main Training Pipeline with:
- Multiclass Classification
- Leave-One-Cohort-Out Cross-Validation
- Nested CV for Hyperparameter Tuning
- Temperature Scaling Calibration
"""
import os
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold

from gcn_model_multiclass import (
    GCN, encode_onehot_multiclass, normalize, sparse_mx_to_torch_sparse_tensor,
    train_epoch, evaluate, get_predictions, get_logits, multiclass_metrics
)
from cohort_cv import (
    LeaveOneCohortOut, extract_cohorts_from_meta, split_validation_from_training
)
from hyperparameter_tuning import nested_cv_random_search
from calibration import TemperatureScaling, compute_ece, compute_mce
from calibration_plot import plot_calibration, save_calibration_metrics


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model_with_best_params(features, adj, labels, train_idx, val_idx,
                                   best_params, n_classes, seed=42):
    """
    Train GCN model with best hyperparameters.

    Args:
        features: Node features
        adj: Adjacency matrix
        labels: Node labels
        train_idx: Training indices
        val_idx: Validation indices (for calibration)
        best_params: Best hyperparameter configuration
        n_classes: Number of classes
        seed: Random seed

    Returns:
        Trained model
    """
    setup_seed(seed)

    model = GCN(
        nfeat=features.shape[1],
        nhid=best_params['hidden_dim'],
        nclass=n_classes,
        dropout=best_params['dropout']
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(best_params['epochs']):
        # Train
        loss_train = train_epoch(model, optimizer, features, adj, labels, train_idx)

        # Validate
        if epoch % 10 == 0 or epoch == best_params['epochs'] - 1:
            loss_val, acc_val, _ = evaluate(model, features, adj, labels, val_idx)
            print(f"  Epoch {epoch+1}/{best_params['epochs']}: "
                  f"Train Loss={loss_train:.4f}, Val Loss={loss_val:.4f}, "
                  f"Val Acc={acc_val:.4f}")

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_model_state = model.state_dict().copy()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def calibrate_model(model, features, adj, labels, cal_idx, n_classes):
    """
    Calibrate model using temperature scaling.

    Args:
        model: Trained GCN model
        features: Node features
        adj: Adjacency matrix
        labels: Node labels
        cal_idx: Calibration set indices
        n_classes: Number of classes

    Returns:
        Calibrated temperature scaling model
    """
    # Get logits from model
    logits = get_logits(model, features, adj, cal_idx).cpu()
    cal_labels = labels[cal_idx].cpu()

    # Fit temperature scaling
    temp_model = TemperatureScaling(n_classes)
    temperature = temp_model.fit(logits, cal_labels)

    print(f"  Calibration complete. Optimal temperature: {temperature:.4f}")

    return temp_model


def run_loco_cv_with_nested_hpo(input_fs, meta_file, graph, rdir,
                                  n_hpo_iter=20, n_inner_folds=3, rseed=42):
    """
    Run leave-one-cohort-out cross-validation with nested hyperparameter optimization.

    Args:
        input_fs: Input feature file path
        meta_file: Metadata file path
        graph: Graph file path
        rdir: Results directory
        n_hpo_iter: Number of random search iterations for HPO
        n_inner_folds: Number of inner CV folds for HPO
        rseed: Random seed

    Returns:
        Dictionary with results
    """
    setup_seed(rseed)

    # Create results directory
    os.makedirs(rdir, exist_ok=True)

    # Load data
    print("Loading data...")
    idx_features_labels = np.genfromtxt(input_fs, dtype=np.dtype(str))
    features = idx_features_labels[:, 1:-1].astype(float)
    features = np.array(features)
    labels_raw = idx_features_labels[:, -1]

    # Encode labels
    labels_onehot, classes_dict = encode_onehot_multiclass(labels_raw)
    n_classes = len(classes_dict)
    print(f"Number of classes: {n_classes}")
    print(f"Classes: {list(classes_dict.keys())}")

    # Convert to torch tensors
    features = sp.csr_matrix(features, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels_onehot)[1])

    # Load graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(graph, dtype=np.int32)
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())),
        dtype=np.int32
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # Extract cohort information
    print("Extracting cohort information...")
    cohorts = extract_cohorts_from_meta(meta_file)
    print(f"Unique cohorts: {np.unique(cohorts)}")

    # Setup leave-one-cohort-out CV
    loco = LeaveOneCohortOut()
    n_splits = loco.get_n_splits(features, labels, cohorts)
    print(f"Number of LOCO-CV folds: {n_splits}")

    # Results storage
    all_results = []
    fold_metrics = []

    # Main LOCO-CV loop
    for fold_idx, (outer_train_idx, test_idx) in enumerate(loco.split(features.numpy(), labels.numpy(), cohorts)):
        test_cohort = cohorts[test_idx[0]]
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx + 1}/{n_splits} - Test Cohort: {test_cohort}")
        print(f"Train size: {len(outer_train_idx)}, Test size: {len(test_idx)}")
        print(f"{'='*80}")

        # Split outer train into inner train and calibration
        inner_train_idx, cal_idx = split_validation_from_training(
            outer_train_idx, labels.numpy(), cohorts, val_ratio=0.2, seed=rseed
        )

        print(f"Inner train size: {len(inner_train_idx)}, Calibration size: {len(cal_idx)}")

        # Nested CV for hyperparameter optimization
        print("\n--- Hyperparameter Optimization (Nested CV) ---")
        best_params, best_score, config_scores = nested_cv_random_search(
            features, adj, labels, inner_train_idx, cohorts,
            n_inner_folds=n_inner_folds, n_iter=n_hpo_iter, seed=rseed
        )

        # Train final model with best hyperparameters on full outer train set
        print("\n--- Training Final Model with Best Hyperparameters ---")
        print(f"Using parameters: {best_params}")
        model = train_model_with_best_params(
            features, adj, labels, outer_train_idx, cal_idx,
            best_params, n_classes, seed=rseed
        )

        # Get uncalibrated predictions
        print("\n--- Getting Uncalibrated Predictions ---")
        test_probs_uncal = get_predictions(model, features, adj, test_idx).cpu().numpy()
        cal_probs_uncal = get_predictions(model, features, adj, cal_idx).cpu().numpy()

        # Calibrate model
        print("\n--- Calibrating Model (Temperature Scaling) ---")
        temp_model = calibrate_model(model, features, adj, labels, cal_idx, n_classes)

        # Get calibrated predictions
        print("\n--- Getting Calibrated Predictions ---")
        test_logits = get_logits(model, features, adj, test_idx).cpu()
        test_probs_cal = temp_model.predict_proba(test_logits)

        cal_logits = get_logits(model, features, adj, cal_idx).cpu()
        cal_probs_cal = temp_model.predict_proba(cal_logits)

        # Compute metrics
        test_labels = labels[test_idx].numpy()
        cal_labels = labels[cal_idx].numpy()

        # Test set metrics
        test_pred_uncal = np.argmax(test_probs_uncal, axis=1)
        test_pred_cal = np.argmax(test_probs_cal, axis=1)
        test_acc_uncal = np.mean(test_pred_uncal == test_labels)
        test_acc_cal = np.mean(test_pred_cal == test_labels)

        test_ece_uncal = compute_ece(test_probs_uncal, test_labels)
        test_mce_uncal = compute_mce(test_probs_uncal, test_labels)
        test_ece_cal = compute_ece(test_probs_cal, test_labels)
        test_mce_cal = compute_mce(test_probs_cal, test_labels)

        print("\n--- Test Set Results ---")
        print(f"Accuracy (uncalibrated): {test_acc_uncal:.4f}")
        print(f"Accuracy (calibrated):   {test_acc_cal:.4f}")
        print(f"ECE (uncalibrated): {test_ece_uncal:.4f}")
        print(f"ECE (calibrated):   {test_ece_cal:.4f}")
        print(f"MCE (uncalibrated): {test_mce_uncal:.4f}")
        print(f"MCE (calibrated):   {test_mce_cal:.4f}")

        # Plot calibration
        calibration_plot_path = os.path.join(rdir, f'calibration_plot_fold{fold_idx+1}.png')
        metrics_dict = plot_calibration(
            test_probs_uncal, test_probs_cal, test_labels, calibration_plot_path
        )
        metrics_dict['temperature'] = temp_model.temperature.item()
        metrics_dict['test_cohort'] = test_cohort
        metrics_dict['test_acc_uncal'] = test_acc_uncal
        metrics_dict['test_acc_cal'] = test_acc_cal

        # Save metrics to file
        metrics_file = os.path.join(rdir, f'calibration_metrics_fold{fold_idx+1}.txt')
        save_calibration_metrics(metrics_dict, metrics_file)

        # Store results
        fold_result = {
            'fold': fold_idx + 1,
            'test_cohort': test_cohort,
            'best_params': best_params,
            'test_acc_uncal': test_acc_uncal,
            'test_acc_cal': test_acc_cal,
            'ece_uncal': test_ece_uncal,
            'ece_cal': test_ece_cal,
            'mce_uncal': test_mce_uncal,
            'mce_cal': test_mce_cal,
            'temperature': temp_model.temperature.item()
        }
        all_results.append(fold_result)
        fold_metrics.append(metrics_dict)

    # Compute average metrics across folds
    print(f"\n{'='*80}")
    print("FINAL RESULTS - Average Across All Folds")
    print(f"{'='*80}")

    avg_test_acc_uncal = np.mean([r['test_acc_uncal'] for r in all_results])
    avg_test_acc_cal = np.mean([r['test_acc_cal'] for r in all_results])
    avg_ece_uncal = np.mean([r['ece_uncal'] for r in all_results])
    avg_ece_cal = np.mean([r['ece_cal'] for r in all_results])
    avg_mce_uncal = np.mean([r['mce_uncal'] for r in all_results])
    avg_mce_cal = np.mean([r['mce_cal'] for r in all_results])
    avg_temp = np.mean([r['temperature'] for r in all_results])

    print(f"Average Test Accuracy (uncalibrated): {avg_test_acc_uncal:.4f} ± {np.std([r['test_acc_uncal'] for r in all_results]):.4f}")
    print(f"Average Test Accuracy (calibrated):   {avg_test_acc_cal:.4f} ± {np.std([r['test_acc_cal'] for r in all_results]):.4f}")
    print(f"Average ECE (uncalibrated): {avg_ece_uncal:.4f} ± {np.std([r['ece_uncal'] for r in all_results]):.4f}")
    print(f"Average ECE (calibrated):   {avg_ece_cal:.4f} ± {np.std([r['ece_cal'] for r in all_results]):.4f}")
    print(f"Average MCE (uncalibrated): {avg_mce_uncal:.4f} ± {np.std([r['mce_uncal'] for r in all_results]):.4f}")
    print(f"Average MCE (calibrated):   {avg_mce_cal:.4f} ± {np.std([r['mce_cal'] for r in all_results]):.4f}")
    print(f"Average Temperature: {avg_temp:.4f}")

    # Save summary results
    summary_file = os.path.join(rdir, 'final_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("LEAVE-ONE-COHORT-OUT CROSS-VALIDATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Per-Fold Results:\n")
        f.write("-" * 80 + "\n")
        for r in all_results:
            f.write(f"Fold {r['fold']} (Test Cohort: {r['test_cohort']}):\n")
            f.write(f"  Test Accuracy (uncal): {r['test_acc_uncal']:.4f}\n")
            f.write(f"  Test Accuracy (cal):   {r['test_acc_cal']:.4f}\n")
            f.write(f"  ECE (uncal): {r['ece_uncal']:.4f}, ECE (cal): {r['ece_cal']:.4f}\n")
            f.write(f"  MCE (uncal): {r['mce_uncal']:.4f}, MCE (cal): {r['mce_cal']:.4f}\n")
            f.write(f"  Temperature: {r['temperature']:.4f}\n")
            f.write(f"  Best Hyperparameters: {r['best_params']}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("AVERAGE RESULTS:\n")
        f.write("=" * 80 + "\n")
        f.write(f"Average Test Accuracy (uncalibrated): {avg_test_acc_uncal:.4f} ± {np.std([r['test_acc_uncal'] for r in all_results]):.4f}\n")
        f.write(f"Average Test Accuracy (calibrated):   {avg_test_acc_cal:.4f} ± {np.std([r['test_acc_cal'] for r in all_results]):.4f}\n")
        f.write(f"Average ECE (uncalibrated): {avg_ece_uncal:.4f} ± {np.std([r['ece_uncal'] for r in all_results]):.4f}\n")
        f.write(f"Average ECE (calibrated):   {avg_ece_cal:.4f} ± {np.std([r['ece_cal'] for r in all_results]):.4f}\n")
        f.write(f"Average MCE (uncalibrated): {avg_mce_uncal:.4f} ± {np.std([r['mce_uncal'] for r in all_results]):.4f}\n")
        f.write(f"Average MCE (calibrated):   {avg_mce_cal:.4f} ± {np.std([r['mce_cal'] for r in all_results]):.4f}\n")
        f.write(f"Average Temperature: {avg_temp:.4f}\n")

    print(f"\nResults saved to: {summary_file}")

    return {
        'all_results': all_results,
        'avg_test_acc_uncal': avg_test_acc_uncal,
        'avg_test_acc_cal': avg_test_acc_cal,
        'avg_ece_uncal': avg_ece_uncal,
        'avg_ece_cal': avg_ece_cal,
        'avg_mce_uncal': avg_mce_uncal,
        'avg_mce_cal': avg_mce_cal
    }
