import pymol

def visualize_ribbon_pdb(pdb_file):
    # Launch PyMOL and create a PyMOL instance
    pymol.finish_launching()
    cmd = pymol.cmd

    # Load the PDB file
    cmd.load(pdb_file, "molecule")

    # Show the ribbon representation
    # cmd.show("ribbon", "molecule")

    # Zoom to the molecule
    # cmd.zoom("molecule")

    # Display the PyMOL viewer
    cmd.show("cartoon", "molecule")

    # Keep the PyMOL viewer open
    pymol.cmd.rock()

# Replace 'output.pdb' with the actual path to your PDB file
visualize_ribbon_pdb('new.pdb')
# visualize_ribbon_pdb('old.pdb')
# visualize_ribbon_pdb('true.pdb')
