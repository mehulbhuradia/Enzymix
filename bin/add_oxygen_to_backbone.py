"""
Script to add oxygen to backbone. This is not strictly included in the original
set of dihedral/bond angles, and it is not technically a part of the backbone,
but is required for some downstream tools to work (e.g. pymol). This script is
meant to be run directly on generated backbones, that have not yet have side
chains added onto them. This also means that if it is run on a input with side
chains, those side chains will be *discarded* in the output. This behavior on 
full side chains is meant to mimic what would happen if we stripped the side
chains and added an O.

Example usage:
python add_oxygen_to_backbone sampled_pdb sampled_pdb_with_o
"""

import os
import logging
import argparse
import glob

from biotite import structure as struct
from biotite.structure.io.pdb import PDBFile

from tqdm.auto import tqdm

import os
from typing import *

import numpy as np
import torch

N_CA_LENGTH = 1.46  # Check, approxiamtely right
CA_C_LENGTH = 1.54  # Check, approximately right
C_N_LENGTH = 1.34  # Check, approximately right

# Taken from initial coords from 1CRN, which is a THR
N_INIT = np.array([17.047, 14.099, 3.625])
CA_INIT = np.array([16.967, 12.784, 4.338])
C_INIT = np.array([15.685, 12.755, 5.133])


def place_dihedral(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_angle: float,
    bond_length: float,
    torsion_angle: float,
    use_torch: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Place the point d such that the bond angle, length, and torsion angle are satisfied
    with the series a, b, c, d.
    """
    assert a.shape == b.shape == c.shape
    assert a.shape[-1] == b.shape[-1] == c.shape[-1] == 3

    if not use_torch:
        unit_vec = lambda x: x / np.linalg.norm(x, axis=-1)
        cross = lambda x, y: np.cross(x, y, axis=-1)
    else:
        ensure_tensor = (
            lambda x: torch.tensor(x, requires_grad=False).to(a.device)
            if not isinstance(x, torch.Tensor)
            else x.to(a.device)
        )
        a, b, c, bond_angle, bond_length, torsion_angle = [
            ensure_tensor(x) for x in (a, b, c, bond_angle, bond_length, torsion_angle)
        ]
        unit_vec = lambda x: x / torch.linalg.norm(x, dim=-1, keepdim=True)
        cross = lambda x, y: torch.linalg.cross(x, y, dim=-1)

    ab = b - a
    bc = unit_vec(c - b)
    n = unit_vec(cross(ab, bc))
    nbc = cross(n, bc)

    if not use_torch:
        m = np.stack([bc, nbc, n], axis=-1)
        d = np.stack(
            [
                -bond_length * np.cos(bond_angle),
                bond_length * np.cos(torsion_angle) * np.sin(bond_angle),
                bond_length * np.sin(torsion_angle) * np.sin(bond_angle),
            ],
            axis=a.ndim - 1,
        )
        d = m.dot(d)
    else:
        m = torch.stack([bc, nbc, n], dim=-1)
        d = torch.stack(
            [
                -bond_length * torch.cos(bond_angle),
                bond_length * torch.cos(torsion_angle) * torch.sin(bond_angle),
                bond_length * torch.sin(torsion_angle) * torch.sin(bond_angle),
            ],
            dim=a.ndim - 1,
        ).type(m.dtype)
        d = torch.matmul(m, d).squeeze()

    return d + c



logging.basicConfig(level=logging.INFO)


def read_structure(fname: str) -> struct.AtomArray:
    """Return an atom array from the given pdb file."""
    with open(fname) as source:
        pdb_file = PDBFile.read(source)
    assert pdb_file.get_model_count() == 1
    structure = pdb_file.get_structure()[0]
    if struct.get_residue_count(structure) != len(structure) // 3:
        logging.warning(f"{fname} contains side-chains, which are discarded")
        structure = structure[struct.filter_backbone(structure)]
    return structure


def add_oxygen_to_backbone(structure: struct.AtomArray) -> struct.AtomArray:
    """Returns a new atom array with oxygen atoms added to the backbone."""
    assert len(structure) % 3 == 0
    assert struct.get_residue_count(structure) == len(structure) // 3

    retval = []
    for i, atom in enumerate(structure):
        atom.atom_id = len(retval) + 1
        atom.res_id = i // 3
        atom.res_name = "GLY"  # Since we are doing backbone only
        retval.append(atom)
        # Last atom in residue after (0, N), (1, CA), (2, C)
        if i % 3 == 2 and i + 1 < len(structure):
            # Insert oxygen
            psi = struct.dihedral(
                structure[i - 2].coord,
                structure[i - 1].coord,
                structure[i].coord,
                structure[i + 1].coord,
            )
            oxy = struct.Atom(
                coord=place_dihedral(
                    retval[-3].coord,
                    retval[-2].coord,
                    retval[-1].coord,
                    torsion_angle=psi,
                    bond_angle=2.0992622,
                    bond_length=1.2359372,
                ),
                chain_id=retval[-1].chain_id,
                res_id=retval[-1].res_id,
                atom_id=len(retval) + 1,
                res_name=retval[-1].res_name,
                atom_name="O",
                element="O",
            )
            # Propogate any other annotations
            for k in retval[-1]._annot.keys():
                if k not in oxy._annot:
                    oxy._annot[k] = retval[-1]._annot[k]
            retval.append(oxy)
    return struct.array(retval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", type=str, help="Input file, or directory with .pdb files"
    )
    parser.add_argument("outdir", type=str, help="Output directory to write .pdb files")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        pdb_files = list(glob.glob(os.path.join(args.input, "*.pdb")))
        logging.info(f"Found {len(pdb_files)} pdb files in {args.input}")
    elif os.path.isfile(args.input):
        pdb_files = [args.input]
    else:
        raise ValueError(f"Invalid input: {args.input}")

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    for fname in tqdm(pdb_files):
        structure = read_structure(fname)
        updated_backbone_with_o = add_oxygen_to_backbone(structure)
        outname = os.path.join(args.outdir, os.path.basename(fname))
        with open(outname, "w") as sink:
            pdb_file = PDBFile()
            pdb_file.set_structure(updated_backbone_with_o)
            pdb_file.write(sink)
        del pdb_file


if __name__ == "__main__":
    main()
