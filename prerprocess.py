import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities
from rdkit import Chem
import pickle
from tqdm import tqdm
import warnings

from protein_ligand import PDBProtein, parse_sdf_file

data_path = "./data/crossdocked/crossdocked_pocket10/crossdocked_pocket10/"

index_path = './data/crossdocked/crossdocked_pocket10/index.pkl'
with open(index_path, 'rb') as f:
    index = pickle.load(f)

sample_protein_path = data_path+"5HT2B_HUMAN_27_404_0/4ib4_A_rec_4ib4_erm_lig_tt_docked_0_pocket10.pdb"
sample_protein_dict = PDBProtein(sample_protein_path).to_dict_atom()

print(sample_protein_dict)

sample_ligand_path = data_path+'5HT2B_HUMAN_27_404_0/4ib4_A_rec_4ib4_erm_lig_tt_docked_0.sdf'
sample_ligand_dict = parse_sdf_file(sample_ligand_path)

print(sample_ligand_dict)

filtered_index = []
for pl in index:
    if(pl[3]<=0.5): 
        if(None not in pl): 
            filtered_index.append(pl)
print(len(filtered_index))

protein_z_id = {1:0, 6:1, 7:2, 8:3, 16:4, 34:5}
protein_z = {1,6,7,8,16,34}
max_aa = 20

def protein_dict_to_feature(dict):

    n = len(dict['element'])
    element = np.zeros((n, len(protein_z)), dtype=np.float32)
    for i,z in enumerate(dict['element']):
        element[i,protein_z_id[z]] = 1
    aminoacid = np.zeros((n, max_aa), dtype=np.float32)
    for i,aa in enumerate(dict['atom_to_aa_type']):
        aminoacid[i, aa] = 1
    backbone = dict['is_backbone'].reshape(n,-1)
    features = np.concatenate((element, aminoacid, backbone), axis=1)
    return features

sample_protein_features = protein_dict_to_feature(sample_protein_dict)
print(sample_protein_features[:3])

ligand_z_id = {1:0,6:1,7:2,8:3,9:4,15:5,16:6,17:7}
ligand_z = [1,6,7,8,9,15,16,17]

def ligand_dict_to_feature(dict):

    n = len(dict['element'])
    element = np.zeros((n, len(ligand_z)), dtype=np.float32)
    for i,z in enumerate(dict['element']):
        element[i,ligand_z_id[z]] = 1
    features = np.concatenate((element, dict['atom_feature']), axis=1)
    return features

sample_ligand_features = ligand_dict_to_feature(sample_ligand_dict)
print(sample_ligand_features[:5])

ligand_z_id_1 = {6:0,7:1,8:2,9:3}
ligand_z_1 = [6,7,8,9]

def ligand_dict_to_feature_1(dict):

    n = len(dict['element'])
    element = np.zeros((n, len(ligand_z_1)), dtype=np.float32)
    for i,z in enumerate(dict['element']):
        if(z not in ligand_z_id_1): return None
        element[i,ligand_z_id_1[z]] = 1
    features = np.concatenate((element, dict['atom_feature']), axis=1)
    return features

sample_ligand_features = ligand_dict_to_feature(sample_ligand_dict)
print(sample_ligand_features[:5])

ligand_z_id_2 = {6:0,7:1,8:2,15:3,16:4,17:5}
ligand_z_2 = [6,7,8,15,16,17]

def ligand_dict_to_feature_2(dict):

    n = len(dict['element'])
    element = np.zeros((n, len(ligand_z_2)), dtype=np.float32)
    for i,z in enumerate(dict['element']):
        if(z not in ligand_z_id_2): return None
        element[i,ligand_z_id_2[z]] = 1
    features = element
    return features

sample_ligand_features = ligand_dict_to_feature_2(sample_ligand_dict)
print(sample_ligand_features[:5])

x = 0
y = 0
res = []
processed_index = []

for ix in tqdm(range(len(filtered_index))):
    
    protein_ligand = index[ix]
    if(None in protein_ligand): continue

    protein_path = data_path + protein_ligand[0]
    ligand_path = data_path + protein_ligand[1]

    protein_data = PDBProtein(protein_path).to_dict_atom()
    ligand_data = parse_sdf_file(ligand_path)
    
    if(ligand_data is None):
        print(ix)
        x += 1
        continue

    protein_feature = protein_dict_to_feature(protein_data)
    ligand_feature = ligand_dict_to_feature_2(ligand_data)

    if(ligand_feature is None):
        y += 1
        continue

    dat = {}
    dat['p'] = protein_ligand[0]
    dat['l'] = protein_ligand[1]
    dat['pf'] = protein_feature 
    dat['lf'] = ligand_feature
    dat['pp'] = protein_data['pos']
    dat['lp'] = ligand_data['pos']

    with open('./data/processed_2/'+str(ix)+'.pickle', 'wb') as f:
        # print("********")
        pickle.dump(dat, f)

    processed_index.append(str(ix)+'.pickle')


with open('./data/processed_2/index.pickle', 'wb') as f:
    pickle.dump(processed_index, f)

print(x,y,x+y, 'not processed')

# ix = 9390
# protein_ligand = index[ix]
# if(None not in protein_ligand):
#     protein_path = data_path + protein_ligand[0]
#     ligand_path = data_path + protein_ligand[1]
#     protein_data = PDBProtein(protein_path).to_dict_atom()
#     ligand_data = parse_sdf_file(ligand_path)

with open('./data/processed_2/0.pickle', 'rb') as f:
    spl = pickle.load(f)
print(spl['lf'])

