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

visualize_ribbon_pdb('traj/0.pdb')
# visualize_ribbon_pdb('true.pdb')
# visualize_ribbon_pdb('gen/3.pdb')
# visualize_ribbon_pdb('gen/4.pdb')
# visualize_ribbon_pdb('gen/6.pdb')
# save_image_from_pdb('gen/7.pdb')
# save_image_from_pdb('gen/9.pdb')
# save_image_from_pdb('gen/10.pdb')
# save_image_from_pdb('gen/11.pdb')
# save_image_from_pdb('gen/12.pdb')
# save_image_from_pdb('gen/13.pdb')
# save_image_from_pdb('gen/14.pdb')
# save_image_from_pdb('gen/15.pdb')
# save_image_from_pdb('gen/16.pdb')
# save_image_from_pdb('gen/18.pdb')
# save_image_from_pdb('gen/20.pdb')
# save_image_from_pdb('gen/21.pdb')
# save_image_from_pdb('gen/22.pdb')
# save_image_from_pdb('gen/23.pdb')
# save_image_from_pdb('gen/25.pdb')
# save_image_from_pdb('gen/26.pdb')
# save_image_from_pdb('gen/27.pdb')
# save_image_from_pdb('gen/28.pdb')
# save_image_from_pdb('gen/29.pdb')
# save_image_from_pdb('gen/30.pdb')
# save_image_from_pdb('gen/31.pdb')
    
# save_image_from_pdb('gen/3_true.pdb')
# save_image_from_pdb('gen/4_true.pdb')
# save_image_from_pdb('gen/6_true.pdb')
# save_image_from_pdb('gen/7_true.pdb')
# save_image_from_pdb('gen/9_true.pdb')
# save_image_from_pdb('gen/10_true.pdb')
# save_image_from_pdb('gen/11_true.pdb')
# save_image_from_pdb('gen/12_true.pdb')
# save_image_from_pdb('gen/13_true.pdb')
# save_image_from_pdb('gen/14_true.pdb')
# save_image_from_pdb('gen/15_true.pdb')
# save_image_from_pdb('gen/16_true.pdb')
# save_image_from_pdb('gen/18_true.pdb')
# save_image_from_pdb('gen/20_true.pdb')
# save_image_from_pdb('gen/21_true.pdb')
# save_image_from_pdb('gen/22_true.pdb')
# save_image_from_pdb('gen/23_true.pdb')
# save_image_from_pdb('gen/25_true.pdb')
# save_image_from_pdb('gen/26_true.pdb')
# save_image_from_pdb('gen/27_true.pdb')
# save_image_from_pdb('gen/28_true.pdb')
# save_image_from_pdb('gen/29_true.pdb')
# save_image_from_pdb('gen/30_true.pdb')
# save_image_from_pdb('gen/31_true.pdb')