# [Protein Structure and Sequence Co-Design through Graph Based Generative Diffusion Modeling](https://repository.tudelft.nl/record/uuid:08ddfafd-1eac-4e53-b0fc-837ad39cb6df)

This project implements a graph-based generative diffusion model for the co-design of protein structures and sequences. It was developed as a master's thesis at TU Delft.

![Project Demo](visualisation/output.gif)

## Abstract

Proteins are fundamental biological macromolecules essential for cellular structure, enzymatic catalysis, and immune defense, making the generation of novel proteins crucial for advancements in medicine, biotechnology, and material sciences. This study explores protein design using deep generative models, specifically Denoising Diffusion Probabilistic Models (DDPMs). While traditional methods often focus on either protein structure or sequence design independently, recent trends emphasize a co-design approach addressing both aspects simultaneously. We propose a novel methodology utilizing Equivariant Graph Neural Networks (EGNNs) within the diffusion framework to co-design protein structures and sequences. We modify the EGNN architecture to improve its effectiveness in learning intricate data patterns. Experimental results show that our approach effectively generates high-quality protein sequences, although challenges remain in producing plausible protein backbones and ensuring strong sequence-structure correlation.

## Installation

### Requirements

- Python 3.10
- PyTorch with CUDA 12.1
- PyTorch Geometric
- tqdm
- easydict
- tensorboard
- graphein
- bioconda::tmalign
- seaborn
- [Omegafold](https://github.com/HeliXonProtein/OmegaFold)
- [ProtienMPNN](https://github.com/dauparas/ProteinMPNN)

### Setup Instructions for [DAIC](https://daic.tudelft.nl/)

Change to the project directory:
```bash
cd path/to/your/project
```
Clone the repository:
```bash
git clone https://github.com/mehulbhuradia/Enzymix.git
cd Enzymix
```
Load the required modules:
```bash
module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10
```

Create and activate a new conda environment:
```bash
conda create --prefix ./thesis
conda activate ./thesis
```

Due to limited space, we need to change the conda cache directory to /tmp/:
```bash
conda config --add pkgs_dirs /tmp/
```

Install the required packages
```bash
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install bioconda::tmalign
conda install -c anaconda tqdm
conda install -c conda-forge easydict
conda install -c conda-forge tensorboard
conda install pyg -c pyg
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install packaging
conda install seaborn
pip install graphein --user
```

Download Omegafold and ProteinMPNN:

Clone the repositories and follow the instructions in the respective READMEs to set them up.
```bash
cd ..
git clone https://github.com/dauparas/ProteinMPNN.git
git clone https://github.com/HeliXonProtein/OmegaFold.git
cd Enzymix
```

Update the paths in `bin/omegafold_across_gpus.py` and `bin/pdb_to_residue_protienmpnn.py` to the correct paths of the Omegafold and ProteinMPNN repositories.

For `omegafold_across_gpus.py`. Replace:
```python
cmd = f"CUDA_VISIBLE_DEVICES={gpu} python /tudelft.net/staff-umbrella/DIMA/OmegaFold/main.py {input_fasta} {outdir} --device cuda:0"
```
with your OmegaFold main.py path:
```python
cmd = f"CUDA_VISIBLE_DEVICES={gpu} python path/to/your/project/OmegaFold/main.py {input_fasta} {outdir} --device cuda:0"
```
For `bin/pdb_to_residue_protienmpnn.py`. Replace:
```python
PROTEINMPNN_SCRIPT = os.path.expanduser("/tudelft.net/staff-umbrella/DIMA/ProteinMPNN/protein_mpnn_run.py")
```
with your ProteinMPNN protein_mpnn_run.py path:
```python
PROTEINMPNN_SCRIPT = os.path.expanduser("path/to/your/project/ProteinMPNN/protein_mpnn_run.py")
```

## Usage

### Downloading Data
Download the `.pdb` files from AlphaFoldDB:
[Swiss-Prot pdbs](https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v4.tar)

Unzip the files to a folder called `swissprot_pdb_v4`. `swissprot_pdb_v4` should be in the same directory as the project.

`swissprot_pdb_v4` should contain .gz files, unzip these files.

`swissprot_pdb_v4` should contain directories with the `.pdb` files. Delete any files that are not `.pdb` files.

### Filtering Data

We filter the data to remove any proteins that are not suitable for our model. We filter the data based on the following criteria:
Average AlphaFold Confidence Score > 70, Sequence Length > 50, and Sequence Length < 100.

Run the `filter_data.ipynb` notebook to filter the data, and remove any proteins that do not meet the criteria.

### Preprocessing Data

Run the `preprocess_data.py` file to preprocess the data. This will generate json files containing tensors of the protein structures and sequences, which will be used for training the model.

```bash
python preprocess_data.py
```

### Training the Model

### Generating Protein Structures and Sequences

### Analysing Generated Protein Structures and Sequences



## Project Structure

## Features

- Data Download and preprocessing
- Model training and evaluation
- Generating protein structures and sequences
- Analysing generated protein structures and sequences

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Authors

- **M.H. Bhuradia** - *Master's thesis* - [TU Delft](https://www.tudelft.nl/)


## Contributors

- **J.M. Weber** - Pattern Recognition and Bioinformatics - EEMCS (mentor)
- **H. Jamali-Rad** - Pattern Recognition and Bioinformatics - EEMCS (mentor)
- **A.O. Villegas Morcillo** - Pattern Recognition and Bioinformatics - EEMCS (mentor)
- **M.J.T. Reinders** - Pattern Recognition and Bioinformatics - EEMCS (graduation committee member)
- **J.W. Böhmer** - Sequential Decision Making - EEMCS (graduation committee member)


