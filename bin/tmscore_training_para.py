"""
Compute the maximum TM score against training set
"""
# NOTE this is a thin wrapper around function called in sample.py

import logging
import os, sys
import re
import json
from glob import glob
from pathlib import Path
import argparse
from typing import *
import multiprocessing as mp

from tqdm.auto import tqdm

from foldingdiff.datasets import CathCanonicalAnglesDataset
from foldingdiff import tmalign


def compute_training_tm_scores(
    pdb_files: Collection[str],
    train_files,
    outdir: Path,
    nthreads: int = os.getenv('SLURM_CPUS_PER_TASK'),
    start: int = 0,
):
    logging.info(f"Calculating tm scores with {nthreads} threads...")
    add_tm_scores, add_tm_scores_ref = {}, {}
    for i, fname in tqdm(enumerate(pdb_files), total=len(pdb_files)):
        samp_name = os.path.splitext(os.path.basename(fname))[0]
        tm_score, tm_score_ref = tmalign.max_tm_across_refs(
            fname,
            train_files,
            n_threads=nthreads,
        )
        add_tm_scores[samp_name] = tm_score
        add_tm_scores_ref[samp_name] = tm_score_ref
    
    # File paths
    tm_scores_path = outdir / f'tm_scores_{start}.json'
    tm_scores_ref_path = outdir / f'tm_scores_ref_{start}.json'

    # Rewrite the updated dictionaries back to the JSON files
    with open(tm_scores_path, "w") as sink:
        json.dump(add_tm_scores, sink, indent=4)

    with open(tm_scores_ref_path, "w") as sink:
        json.dump(add_tm_scores_ref, sink, indent=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--dirname",
        type=str,
        default=os.path.join(os.getcwd(), "generated/pdb"),
        help="Directory of generated PDB structures",
    )
    parser.add_argument(
        "-n", "--nsubset", type=int, default=0, help="Take only first n hits, 0 ignore"
    )
    parser.add_argument(
        "-t",
        "--train_path",
        type=str,
        default=os.path.join(os.getcwd(), "swiss_prot_pdbs"),
        help="Path to training dataset",
    )
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--chunk', type=int, default=6)

    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    assert os.path.isdir(args.dirname)
    generated_pdbs = glob(os.path.join(args.dirname, "*.pdb"))
    int_extractor = lambda x: tuple(
        [int(i) for i in re.findall(r"[0-9]+", os.path.basename(x))]
    )
    generated_pdbs = sorted(generated_pdbs, key=int_extractor)
    assert generated_pdbs
    logging.info(f"Found {len(generated_pdbs)} generated structures")
    if args.nsubset > 0:
        logging.info(f"Subsetting to the first {args.nsubset} pdb files")
        generated_pdbs = generated_pdbs[: args.nsubset]

    train_files = glob(os.path.join(args.train_path, "*.pdb"))
    
    # Calculate scores
    compute_training_tm_scores(generated_pdbs[args.start:args.start+args.chunk], train_files, Path(args.dirname),start=args.start)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
