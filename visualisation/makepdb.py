from Bio.PDB import Atom, Model, Chain, Residue, Structure, PDBIO

def create_pdb_file(data, output_filename,onlyca=False):
    # Create a Biopython Structure
    structure = Structure.Structure('example_structure')

    # Create a Model
    model = Model.Model(0)

    # Create a Chain
    chain = Chain.Chain('A')

    # Add atoms to the chain
    for index, entry in enumerate(data, start=1):
        residue = Residue.Residue((' ', index, ' '), entry['name'], index)
        if not onlyca:
            residue.add(Atom.Atom('N', entry['N'], 0, 0, ' ', 'N', index, 'N'))
            residue.add(Atom.Atom('C', entry['C'], 0, 0, ' ', 'C', index, 'C'))
        residue.add(Atom.Atom('CA', entry['CA'], 0, 0, ' ', 'CA', index, 'C'))
        chain.add(residue)

    # Add the chain to the model
    model.add(chain)

    # Add the model to the structure
    structure.add(model)

    # Save the structure to a PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_filename)

# Example usage:
data = [
    {'name': 'PYL', 'CA': [-1.5914127826690674, 0.6234506368637085, 0.5534527897834778],
     'CB': [-0.5910303592681885, -0.23990178108215332, -0.29048001766204834],
     'CN': [1.1026897430419922, 0.4725823700428009, 1.9468915462493896]},
    {'name': 'GLY', 'CA': [0.40310436487197876, 0.12407051771879196, -1.0085489749908447],
     'CB': [0.41473421454429626, -0.5919309258460999, 0.03967459127306938],
     'CN': [-1.449580430984497, 0.5406938791275024, 0.16636748611927032]},
    {'name': 'ILE', 'CA': [-0.5803375840187073, -2.8976047039031982, -0.9644773602485657],
     'CB': [0.640600860118866, 1.2718552350997925, 0.9594387412071228],
     'CN': [0.2503027021884918, 1.2377198934555054, -1.3929052352905273]},
    {'name': 'THR', 'CA': [0.17565006017684937, 1.5071712732315063, 0.04689114913344383],
     'CB': [-0.8249223232269287, 0.018620770424604416, 1.4885622262954712],
     'CN': [-1.8985326290130615, 0.276487797498703, 0.27655795216560364]},
    {'name': 'GLY', 'CA': [0.5908322930335999, 0.8758586645126343, 1.6808959245681763],
     'CB': [-0.02423665300011635, 1.4843624830245972, 0.6956225037574768],
     'CN': [-1.3424124717712402, 2.0843119621276855, 1.5294557809829712]}
]

# create_pdb_file(data, "example.pdb")
