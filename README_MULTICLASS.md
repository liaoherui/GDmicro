# GDmicro - Multiclass Classification with Calibration

## New Features

This enhanced version of GDmicro adds support for:

1. **Multiclass Classification**: Support for more than 2 disease classes (not just binary healthy/disease)
2. **Temperature Scaling Calibration**: Calibrates model predictions using temperature scaling fitted on held-out validation data
3. **Calibration Metrics**: Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
4. **Calibration Plots**: Generates reliability diagrams showing model calibration before and after temperature scaling
5. **Leave-One-Cohort-Out Cross-Validation (LOCO-CV)**: Main cross-validation strategy that leaves out one study/cohort at a time
6. **Nested Cross-Validation for Hyperparameter Optimization**: Uses random search with nested CV to tune:
   - Hidden layer dimensions
   - Learning rate
   - Weight decay
   - Dropout rate
   - Number of epochs
7. **Accuracy-Based Tuning**: Uses accuracy as the performance metric for hyperparameter selection

## Installation

Same as the original GDmicro. Ensure you have the following additional dependencies:

```bash
pip install matplotlib scipy scikit-learn torch numpy pandas
```

## Usage

### Basic Usage

Run multiclass classification with leave-one-cohort-out cross-validation:

```bash
python run_GDmicro_multiclass.py \
    -i <input_file.csv> \
    -d <disease_name> \
    -o <output_directory>
```

### Advanced Options

```bash
python run_GDmicro_multiclass.py \
    -i Input_files/multiclass_data.csv \
    -d IBD \
    -o Results/IBD_multiclass \
    -k 5 \
    -s 42 \
    --n_hpo_iter 30 \
    --n_inner_folds 3
```

### Parameters

- `-i, --input_file`: Path to input CSV file (required)
- `-d, --disease`: Disease name (required)
- `-o, --outdir`: Output directory (default: GDmicro_multiclass_res)
- `-k, --kneighbor`: Number of neighbors in KNN graph (default: 5)
- `-s, --randomseed`: Random seed for reproducibility (default: 42)
- `--n_hpo_iter`: Number of random search iterations for hyperparameter optimization (default: 20)
- `--n_inner_folds`: Number of inner CV folds for HPO (default: 3)

## Input Format

The input format is the same as original GDmicro, but now supports multiple disease classes:

| subject_id | class | disease | study | Acidaminococcus_intestini | ... |
|------------|-------|---------|-------|---------------------------|-----|
| S1_a_WGS | train | IBD_UC | GBR | 0.0 | ... |
| S2_a_WGS | train | IBD_CD | GBR | 0.0 | ... |
| S3_a_WGS | train | Health | GBR | 0.0 | ... |

Where:
- **subject_id**: Sample identifier
- **class**: Must be "train" for training samples
- **disease**: Can now have multiple disease classes (e.g., IBD_UC, IBD_CD, Health, T2D, etc.)
- **study**: Study/cohort identifier (used for leave-one-cohort-out CV)
- Remaining columns: Microbial species abundances

## Output Files

The pipeline generates the following outputs in the results directory:

### Per-Fold Outputs

For each LOCO-CV fold:

- `calibration_plot_fold{N}.png`: Reliability diagram showing calibration before and after temperature scaling
- `calibration_metrics_fold{N}.txt`: Detailed calibration metrics (ECE, MCE, temperature)

### Summary Outputs

- `final_summary.txt`: Complete results across all folds, including:
  - Per-fold test accuracy (uncalibrated and calibrated)
  - Per-fold calibration metrics (ECE, MCE)
  - Best hyperparameters for each fold
  - Average results across all folds with standard deviations

## How It Works

### 1. Outer Loop: Leave-One-Cohort-Out Cross-Validation

For each unique cohort/study:
- Hold out all samples from that cohort as the test set
- Use all samples from other cohorts for training

### 2. Inner Loop: Nested Hyperparameter Optimization

For the training set of each outer fold:
- Split training data into calibration set (20%) and inner training set (80%)
- Perform random search over hyperparameter space
- For each hyperparameter configuration:
  - Evaluate using inner cross-validation (leave-one-cohort-out or k-fold)
  - Compute average accuracy across inner folds
- Select configuration with best average accuracy

### 3. Model Training

- Train final model using best hyperparameters on full training set (minus calibration set)
- Get uncalibrated predictions on test set

### 4. Calibration

- Fit temperature scaling parameter using calibration set
- Apply temperature scaling to test set predictions
- Generate calibration plot and metrics

### 5. Evaluation

Compute and report:
- Test accuracy (before and after calibration)
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Temperature parameter

## Key Differences from Original GDmicro

| Feature | Original GDmicro | Multiclass Version |
|---------|-----------------|-------------------|
| Classification | Binary only | Multiclass support |
| Cross-validation | K-fold | Leave-one-cohort-out |
| Hyperparameter tuning | Fixed parameters | Nested CV with random search |
| Tuning metric | AUC | Accuracy |
| Calibration | No | Temperature scaling |
| Calibration metrics | No | ECE, MCE |
| Calibration plots | No | Reliability diagrams |

## Example Workflow

```bash
# Run multiclass classification on IBD data with UC, CD, and Health classes
python run_GDmicro_multiclass.py \
    -i Input_files/IBD_multiclass.csv \
    -d IBD \
    -o Results/IBD_multiclass \
    -k 5 \
    -s 42 \
    --n_hpo_iter 30 \
    --n_inner_folds 3
```

This will:
1. Load the IBD data with multiple classes
2. Extract cohort information from the 'study' column
3. For each cohort:
   - Hold it out as test set
   - Optimize hyperparameters on remaining cohorts
   - Train model with best hyperparameters
   - Calibrate using temperature scaling
   - Evaluate on test cohort
   - Generate calibration plot
4. Report average results across all cohorts

## Interpreting Results

### Calibration Plots

The calibration plots show two reliability diagrams:

- **Left plot (Before Calibration)**: Shows how well predicted confidences match actual accuracy. Closer to diagonal = better calibrated.
- **Right plot (After Temperature Scaling)**: Shows improved calibration after temperature scaling.

### Calibration Metrics

- **ECE (Expected Calibration Error)**: Average difference between confidence and accuracy across all confidence bins. Lower is better. Well-calibrated models have ECE close to 0.
- **MCE (Maximum Calibration Error)**: Maximum difference between confidence and accuracy across all bins. Lower is better.

### Temperature

- Temperature > 1: Model is overconfident, temperature scaling makes predictions less confident
- Temperature < 1: Model is underconfident (rare)
- Temperature ≈ 1: Model is well-calibrated

## References

**Temperature Scaling:**
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML.

**Leave-One-Cohort-Out CV:**
- Common practice in microbiome studies for assessing generalization across different cohorts/studies

**Original GDmicro:**
- Liao, H., Shang, J., & Sun, Y. (2023). GDmicro: classifying host disease status with GCN and Deep adaptation network based on the human gut microbiome data. Bioinformatics.

## Contact

For questions about the multiclass enhancements, please refer to the original GDmicro contact information.
