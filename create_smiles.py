import numpy as np
from rdkit import Chem
import pickle
from tqdm import tqdm

from protein_ligand import PDBProtein, parse_sdf_file

data_path = "./data/crossdocked/crossdocked_pocket10/crossdocked_pocket10/"

index_path = './data/crossdocked/crossdocked_pocket10/index.pkl'
with open(index_path, 'rb') as f:
    index = pickle.load(f)

filtered_index = []
for pl in index:
    if(pl[3]<=0.5): 
        if(None not in pl): 
            filtered_index.append(pl)
print(len(filtered_index))

all_smiles = ''
# filtered_index = filtered_index[:10]

for ix in tqdm(range(len(filtered_index))):
    
    protein_ligand = index[ix]
    if(None in protein_ligand): continue

    ligand_path = data_path + protein_ligand[1]

    try:
        rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        smiles = Chem.MolToSmiles(rdmol)
        all_smiles += smiles
        all_smiles += ', '
        all_smiles += protein_ligand[1]
        all_smiles += '\n'
    except:
        pass

with open('smiles_2.csv', 'w') as f:
    f.write(all_smiles)
# print(all_smiles)
    

