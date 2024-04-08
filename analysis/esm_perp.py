import pandas as pd
import numpy as np
from pyfaidx import Fasta
import transformers
from transformers import EsmTokenizer, EsmForMaskedLM
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# If we use large scale model(3B, 15B), we can get higher score than 650M.
# facebook/esm2_t48_15B_UR50D
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)


sequences = Fasta('generated_seqs/hope.fasta')
seqs=[]
for seq in sequences:
    seqs.append(seq[:].seq)


print(len(seqs)) #2660

def score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    
    # mask one by one except [CLS] and [SEP]
    mask = torch.ones(tensor_input.size(-1) -1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.no_grad():
        loss = model(masked_input.to(device), labels=labels.to(device)).loss
    return np.exp(loss.item())

ppl = []
ppl_dict=[]
for i in seqs:
    num = score(model=model, tokenizer=tokenizer, sentence = i)
    ppl.append(num)
    ppl_dict.append({i:num})
    print(num)

mean_ppl = np.mean(ppl)

print("mean:",mean_ppl)

import json

# Specify the file path where you want to save the JSON data
json_file_path = 'ppl_hope.json'

# Save the array to a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(ppl_dict, json_file)
