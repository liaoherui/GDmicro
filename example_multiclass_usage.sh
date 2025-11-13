#!/bin/bash

# Example script for running GDmicro Multiclass Classification
# with Leave-One-Cohort-Out Cross-Validation and Calibration

echo "========================================"
echo "GDmicro Multiclass Example"
echo "========================================"

# Example 1: Basic usage with default parameters
echo ""
echo "Example 1: Basic usage"
echo "----------------------------------------"
echo "python run_GDmicro_multiclass.py -i Input_files/your_data.csv -d DISEASE -o Results/test_run"
echo ""

# Example 2: Advanced usage with custom parameters
echo "Example 2: Advanced usage with hyperparameter optimization"
echo "----------------------------------------"
echo "python run_GDmicro_multiclass.py \\"
echo "    -i Input_files/your_data.csv \\"
echo "    -d DISEASE \\"
echo "    -o Results/advanced_run \\"
echo "    -k 5 \\"
echo "    -s 42 \\"
echo "    --n_hpo_iter 30 \\"
echo "    --n_inner_folds 3"
echo ""

# Example 3: Quick test with minimal HPO iterations
echo "Example 3: Quick test (faster, fewer HPO iterations)"
echo "----------------------------------------"
echo "python run_GDmicro_multiclass.py \\"
echo "    -i Input_files/your_data.csv \\"
echo "    -d DISEASE \\"
echo "    -o Results/quick_test \\"
echo "    --n_hpo_iter 5"
echo ""

echo "========================================"
echo "Output Files"
echo "========================================"
echo "After running, check the output directory for:"
echo "  - calibration_plot_fold{N}.png: Calibration plots for each fold"
echo "  - calibration_metrics_fold{N}.txt: Calibration metrics for each fold"
echo "  - final_summary.txt: Summary of all results"
echo ""

echo "========================================"
echo "Notes"
echo "========================================"
echo "- Input file must have 'study' column for cohort information"
echo "- 'disease' column can have multiple classes (not just binary)"
echo "- The script automatically handles:"
echo "    * Leave-one-cohort-out cross-validation"
echo "    * Nested hyperparameter optimization"
echo "    * Temperature scaling calibration"
echo "    * Calibration metrics (ECE, MCE)"
echo ""
