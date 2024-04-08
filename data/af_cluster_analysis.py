import csv

file_path = "./extras/1-AFDBClusters-entryId_repId_taxId.tsv"

# Open the TSV file in read mode with appropriate encoding
with open(file_path, 'r', encoding='utf-8') as tsvfile:
    # Use csv reader to read the TSV file, specifying the delimiter as '\t' (tab)
    tsvreader = csv.reader(tsvfile, delimiter='\t')
    
    # Iterate through each row in the TSV file
    for row in tsvreader:
        # Print each row
        print(row)
        break