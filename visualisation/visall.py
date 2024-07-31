import pymol
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=str)
args = parser.parse_args()

def save_image_from_pdb(input_pdb):
    # Launch PyMOL and create a PyMOL instance in command-line mode
    pymol.finish_launching(['pymol', '-c'])

    # Load the PDB file
    cmd = pymol.cmd
    cmd.load(input_pdb, "molecule")

    # Show the ribbon representation
    cmd.show("cartoon", "molecule")

    output_image = input_pdb.replace('.pdb', '.png')

    # Save the image
    cmd.png(output_image, width=800, height=600, dpi=300)

    # Quit PyMOL
    cmd.quit()

input_pdb = f'traj/{args.n}.pdb'

save_image_from_pdb(input_pdb)