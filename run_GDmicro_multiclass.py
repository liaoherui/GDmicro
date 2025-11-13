"""
GDmicro Multiclass Training Wrapper
Integrates with existing GDmicro preprocessing pipeline
"""
import os
import sys
import argparse
import numpy as np
from GDmicro_preprocess import preprocess
from train_multiclass_loco_cv import run_loco_cv_with_nested_hpo
import run_MLP_embedding_train_mode


def build_dir(inp):
    """Create directory if it doesn't exist."""
    if not os.path.exists(inp):
        os.makedirs(inp)


def run_multiclass_pipeline(input_file, disease, outdir, kneighbor=5,
                              rseed=42, n_hpo_iter=20, n_inner_folds=3):
    """
    Run complete multiclass GDmicro pipeline with:
    - Leave-one-cohort-out cross-validation
    - Nested CV for hyperparameter optimization
    - Temperature scaling calibration
    - ECE/MCE metrics and calibration plots

    Args:
        input_file: Input CSV file path
        disease: Disease name
        outdir: Output directory
        kneighbor: Number of neighbors for KNN graph
        rseed: Random seed
        n_hpo_iter: Number of HPO iterations (random search)
        n_inner_folds: Number of inner CV folds for HPO

    Returns:
        Results dictionary
    """
    print("\n" + "="*80)
    print("GDmicro - Multiclass Classification with Calibration")
    print("Leave-One-Cohort-Out Cross-Validation with Nested HPO")
    print("="*80 + "\n")

    # Preprocess data
    print("Step 1: Preprocessing data...")
    indir, oin = preprocess(input_file, train_mode=1, disease=disease, out=outdir)

    # Setup paths
    input_fs = f"{indir}/{disease}_sp_train_norm_node.csv"
    eg_fs_norm = f"{indir}/{disease}_sp_train_raw_node.csv"
    meta = f"{indir}/{disease}_meta.tsv"
    insp = f"{indir}/{disease}_train_sp_norm.csv"

    # Create output directories
    gdir = f"{outdir}/Graph_File"
    rdir = f"{outdir}/Res_File"
    build_dir(gdir)
    build_dir(rdir)

    # Check if files exist
    if not os.path.exists(input_fs):
        print(f"Error: Input file not found: {input_fs}")
        sys.exit(1)
    if not os.path.exists(meta):
        print(f"Error: Metadata file not found: {meta}")
        sys.exit(1)

    # Build graph using existing MLP embedding
    print("\nStep 2: Building graph with MLP embeddings...")

    # We need to determine train and validation indices
    # For the graph building, we'll use all training data initially
    # The actual CV splits will be done in the main pipeline

    # Load metadata to get indices
    tid2name = {}
    fm = open(meta, 'r')
    fm.readline()  # Skip header
    c = 0
    train_id = 0
    while True:
        line = fm.readline().strip()
        if not line:
            break
        ele = line.split()
        tid2name[c] = ele[2]
        train_id = c
        c += 1
    train_id += 1
    fm.close()

    # For graph building, use a simple 80-20 split initially
    # The actual LOCO-CV will handle the proper splits
    train_idx = list(range(int(0.8 * train_id)))
    val_idx = list(range(int(0.8 * train_id), train_id))

    # Build graph (this creates a single graph for the whole dataset)
    print("Building KNN graph...")
    graph = run_MLP_embedding_train_mode.build_graph_mlp(
        insp, train_idx, val_idx, meta, disease, fold=1,
        gdir=gdir, kneighbor=kneighbor, rseed=rseed, rdir=rdir
    )

    print(f"Graph created: {graph}")

    # Run main LOCO-CV pipeline with nested HPO
    print("\nStep 3: Running Leave-One-Cohort-Out CV with Nested HPO...")
    results = run_loco_cv_with_nested_hpo(
        input_fs=input_fs,
        meta_file=meta,
        graph=graph,
        rdir=rdir,
        n_hpo_iter=n_hpo_iter,
        n_inner_folds=n_inner_folds,
        rseed=rseed
    )

    print("\n" + "="*80)
    print("Pipeline Complete!")
    print(f"Results saved to: {rdir}")
    print("="*80 + "\n")

    return results


def main():
    """Main entry point."""
    usage = "GDmicro Multiclass - Multiclass classification with calibration and LOCO-CV"
    parser = argparse.ArgumentParser(prog="run_GDmicro_multiclass.py", description=usage)

    parser.add_argument('-i', '--input_file', dest='input_file', type=str, required=True,
                        help="The directory of the input csv file.")
    parser.add_argument('-d', '--disease', dest='disease', type=str, required=True,
                        help="The name of the disease.")
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, default="GDmicro_multiclass_res",
                        help="Output directory of results. (Default: GDmicro_multiclass_res)")
    parser.add_argument('-k', '--kneighbor', dest='kneighbor', type=int, default=5,
                        help="The number of neighborhoods in the knn graph. (default: 5)")
    parser.add_argument('-s', '--randomseed', dest='rseed', type=int, default=42,
                        help="The random seed. (default: 42)")
    parser.add_argument('--n_hpo_iter', dest='n_hpo_iter', type=int, default=20,
                        help="Number of hyperparameter optimization iterations (default: 20)")
    parser.add_argument('--n_inner_folds', dest='n_inner_folds', type=int, default=3,
                        help="Number of inner CV folds for HPO (default: 3)")

    args = parser.parse_args()

    # Run pipeline
    results = run_multiclass_pipeline(
        input_file=args.input_file,
        disease=args.disease,
        outdir=args.outdir,
        kneighbor=args.kneighbor,
        rseed=args.rseed,
        n_hpo_iter=args.n_hpo_iter,
        n_inner_folds=args.n_inner_folds
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
