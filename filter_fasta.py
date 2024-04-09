from Bio import SeqIO

# Replace 'input.fasta' with the path to your input FASTA file
input_file = "/tudelft.net/staff-umbrella/DIMA/Enzymix/uniprot_trembl.fasta"
# Replace 'output.fasta' with the desired output file path
output_file = "smallaf50.fasta"

# Open input and output files
with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    # Iterate through the input FASTA file
    for record in SeqIO.parse(f_in, "fasta"):
        # Check if the sequence length is less than 100
        if len(record.seq) <= 100:
            # Write the record to the output FASTA file
            SeqIO.write(record, f_out, "fasta")
