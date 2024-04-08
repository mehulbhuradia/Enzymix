import os
import glob
import re
import warnings
import multiprocessing as mp
import argparse
import tempfile
from pathlib import Path
from typing import *

import biotite.structure as struc
import biotite.structure.io as strucio
from biotite.structure.io.pdb import PDBFile
import imageio
import pymol

from tqdm.auto import tqdm

def annot_ss_psea(fname: str):
    """Determine secondary structure using PSEA and set with pymol"""
    # https://kpwu.wordpress.com/2011/10/06/pymol-assign-secondary-structural-regions/
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    source = PDBFile.read(fname)
    assert source.get_model_count() == 1
    source_struct = source.get_structure()[0]

    # Get chain ID
    chain_ids = set(source_struct.chain_id)
    assert len(chain_ids) == 1
    chain_id = chain_ids.pop()

    # Get secondary structure
    ss = struc.annotate_sse(source_struct, chain_id)

    for i, s in enumerate(ss):
        if s == "a":  # Helix
            pymol.cmd.alter(f"resi {i}-{i}/", "ss='H'")
        elif s == "b":  # Sheet
            pymol.cmd.alter(f"resi {i}-{i}/", "ss='S'")
    pymol.cmd.rebuild()



def pdb2png(
    pdb_fname: str
) -> str:
    """Convert the pdb file into a png, returns output filename"""
    # https://gist.github.com/bougui505/11401240
    png_fname = pdb_fname.replace(".pdb", ".png")
    assert png_fname.endswith(".png")
    pymol.cmd.load(pdb_fname)
    pymol.cmd.show("cartoon")
    pymol.cmd.color("blue")
    pymol.cmd.set("cartoon_tube_radius", 0.2)
    pymol.cmd.set("ray_opaque_background", 0)
    pymol.cmd.png(png_fname, ray=1, dpi=800)
    pymol.cmd.delete("*")  # So we dont' draw multiple images at once
    return png_fname

pdb2png("./generated/100/pdb/0_0.pdb")
pdb2png("./generated/100/pdb/1_0.pdb")
pdb2png("./generated/100/pdb/2_0.pdb")
pdb2png("./generated/100/pdb/3_0.pdb")


