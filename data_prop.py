from tqdm import tqdm
# import sys
import rdkit
from rdkit.Chem import Descriptors,Draw,QED #,AddHs,MolToSmiles,SDWriter
import matplotlib.pyplot as plt
import seaborn as sns

import sascorer
from utils import load_file #, find_nearest_protein_atoms_to_ligand, get_action_points
from reconstruct import reconstruct_from_struct

ligand_z_id = {6:0,7:1,8:2,15:3,16:4,17:5}
ligand_z = [6,7,8,15,16,17]
NO_ATOM = len(ligand_z)

data_path = '../../data/processed_2/'

index = load_file(data_path+'index.pickle')

logps = []
qeds = []
sascores = []

failed = 0

for i in tqdm(range(len(index))):

    try:

        file_name = data_path+index[i]
        protein_ligand_pair = load_file(file_name)
        n_ligand = len(protein_ligand_pair['lp'])

        # pp = (protein_ligand_pair['pp'])
        # pf = (protein_ligand_pair['pf'])
        lp = (protein_ligand_pair['lp'])
        lf = (protein_ligand_pair['lf'])

        rdmol = reconstruct_from_struct(lp, lf)

        rdkit.Chem.rdDepictor.GenerateDepictionMatching3DStructure(rdmol, rdmol)
        # Draw.MolToFile(rdmol, './outputs/'+out_name+'/_'+str(i)+'.o.png')
        pilimage = Draw.MolToImage(rdmol)
        # print(MolToSmiles(rdmol))

        # rdmol = AddHs(rdmol)
        # logp and qed are the same irrespective of hydrogen addition

        logp = (Descriptors.MolLogP(rdmol))
        qed = QED.qed(rdmol)
        # props = QED.properties(rdmol)
        sascore = sascorer.calculateScore(rdmol)

        logps.append(logp)
        qeds.append(qed)
        sascores.append(sascore)
    
    except:
        
        failed += 1

print(failed)

plt.figure(figsize=(20,20))

plt.subplot(3,1,1)
plt.xlabel('logP', labelpad=10)
sns.histplot(data=logps, kde=True)

plt.subplot(3,1,2)
plt.xlabel('QED', labelpad=10)
sns.histplot(data=qeds, kde=True)

plt.subplot(3,1,3)
plt.xlabel('SA Score', labelpad=10)
sns.histplot(data=sascores, kde=True)

plt.savefig('./logs/data_props.png')