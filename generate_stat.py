import torch
from tqdm import tqdm
import shutil
import numpy as np
import time
import os
import rdkit
from rdkit.Chem import Descriptors,Draw,QED,AddHs,MolToSmiles,SDWriter
import matplotlib.pyplot as plt
import seaborn as sns
# from pymol import cmd
from PIL import Image, ImageDraw, ImageFont

import sascorer
from distance_predictor import DistancePredictor
from type_predictor import TypePredictor
from utils import load_file, find_nearest_protein_atoms_to_ligand, get_action_points
from reconstruct import reconstruct_from_struct

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

protein_z_id = {1:0, 6:1, 7:2, 8:3, 16:4, 34:5}
protein_z = {1,6,7,8,16,34}
max_aa = 20

ligand_z_id = {6:0,7:1,8:2,15:3,16:4,17:5}
ligand_z = [6,7,8,15,16,17]
NO_ATOM = len(ligand_z)

data_path = '../../data/processed_2/'

index = load_file(data_path+'index.pickle')
sample_protein_ligand = load_file(data_path+index[0])

pfdim = len(sample_protein_ligand['pf'][0])
lfdim = len(sample_protein_ligand['lf'][0])

actor = TypePredictor(
    num_classes=len(ligand_z),
    hidden_channels=64,
    protein_atom_feature_dim=pfdim,
    ligand_atom_feature_dim=lfdim,
    aggregate_k = 10
)
actor.to(device)

distance_model = DistancePredictor(
    hidden_channels=128,
    protein_atom_feature_dim=pfdim,
    ligand_atom_feature_dim=lfdim
)
distance_model.to(device)

try:
    distance_model.load_state_dict(torch.load('distance_predictor_checkpoint.pt')['model_state_dict'])
    print('Distance Model Loaded')
except:
    print("Saved Distance Model Not Found")

try:
    actor.load_state_dict(torch.load('model_checkpoint.pt')['actor_state_dict'])
    print('Model Loaded')
except:
    print('No saved model found')

limits = [1.3, 1.8]
resolution = 0.05

try: os.mkdir("./outputs_1/mxQED")
except: pass

p_path = ''
l_path = ''

def generate(ix):

    global p_path, l_path

    out_name = time.strftime("%Y%m%d%H%M%S")+'_'+str(ep)
    os.mkdir("./outputs_1/"+out_name)

    i = 0; stat = [0]*(1+lfdim)
    logp = -10; qed = -1; sascore = -1; mxqed = -1; mxi = 0; mxlogp = -10; mxsa = -1 

    file_name = data_path+index[np.random.randint(len(index))]
    protein_ligand_pair = load_file(file_name)
    n_ligand = len(protein_ligand_pair['lp'])

    if(n_ligand <= 3): return False
    
    pp = torch.from_numpy(protein_ligand_pair['pp']).to(device)
    pf = torch.from_numpy(protein_ligand_pair['pf']).to(device)
    lp = torch.from_numpy(protein_ligand_pair['lp']).to(device)
    lf = torch.from_numpy(protein_ligand_pair['lf']).to(torch.float32).to(device)

    # ctx_protein = find_nearest_protein_atoms_to_ligand(pp, lp, k = 75)
    ctx_pp = pp # [ctx_protein]
    ctx_pf = pf # [ctx_protein]
    ctx_lp = lp[:3] # None # if (rand_len == 0) else lp[:rand_len]
    ctx_lf = lf[:3] # None # if (rand_len == 0) else lf[:rand_len]

    print(lf.shape[0])
    
    centers = torch.cat([ctx_pp, ctx_lp]) # if (rand_len == 0) else torch.cat([ctx_pp, ctx_lp])
    
    done = False

    while(not done):
    
        action_points = get_action_points(ctx_lp, centers, limits, resolution)

        with torch.no_grad():

            dists = distance_model(
                protein_pos = ctx_pp,
                protein_atom_feature = ctx_pf,
                ligand_pos = ctx_lp,
                ligand_atom_feature = ctx_lf
            )

            action_points_dist = torch.cdist(action_points, ctx_lp)
            action_points_dist_diff = torch.cdist(action_points_dist, dists.view(1,-1))
            min_err_ix = torch.argmin(action_points_dist_diff)
            max_density_loc = action_points[min_err_ix]

            atom_cls, terminate = actor(
                pos_query=max_density_loc.view(1,-1),
                protein_pos=ctx_pp, 
                protein_atom_feature=ctx_pf, 
                ligand_pos=ctx_lp, 
                ligand_atom_feature=ctx_lf
            )
        
        done = (terminate.item() > 0.7) or (i>50); (3+i)>=lf.shape[0];  stat[len(ligand_z)] = terminate.item()
        if(done):
            print('Terminate    :', terminate.item()) 
            stats.set_description_str('Stat:'+str(stat))
            continue

        type_next = torch.argmax(atom_cls) # torch.argmax(lf[3+i])
        stat[type_next] += 1; stats.set_description_str('Stat:'+str(stat))
        # if(stat[3] >= 3): break
        pos_next = max_density_loc # lp[3+i]
        i += 1
        
        new_atom_feature = torch.zeros((1, lfdim)).to(device)
        new_atom_feature[0,type_next] = 1

        if(ctx_lp is None):
            ctx_lp = torch.cat([pos_next.view(1,3)])
            ctx_lf = torch.cat([new_atom_feature])
        else:
            ctx_lp = torch.cat([ctx_lp, pos_next.view(1,3)])
            ctx_lf = torch.cat([ctx_lf, new_atom_feature])
            
        centers = torch.cat([centers, pos_next.view(1,3)])

        if(log_vals):
            print("#            :", i)
            print('Dist         :', atom_cls)
            print("Chosen       :", type_next)
            print("Terminate    :", terminate)
            print("Position     :", pos_next)
            print('-'*60)

        res = {}
        res['p'] = protein_ligand_pair['p']
        res['lp'] = ctx_lp.to('cpu').numpy()
        res['lf'] = ctx_lf.to('cpu').numpy()

        # print(res['p'])
        
        try:
            rdmol = reconstruct_from_struct(res['lp'], res['lf'])
            rdmol_og = rdmol

            with SDWriter('./outputs_1/'+out_name+'/'+str(i)+'.sdf') as w:
                w.write(rdmol)

            # cmd.reinitialize()
            # cmd.load('./outputs_1/'+out_name+'/'+str(i)+'.sdf')
            # cmd.png('./outputs_1/'+out_name+'/'+str(i)+'.png')
            
            rdkit.Chem.rdDepictor.GenerateDepictionMatching3DStructure(rdmol, rdmol)
            # Draw.MolToFile(rdmol, './outputs_1/'+out_name+'/_'+str(i)+'.o.png')
            pilimage = Draw.MolToImage(rdmol)
            # torch.save(res, './outputs_1/'+out_name+'/'+str(i)+'.pt')
            # print(pilimage)

            # print(MolToSmiles(rdmol))

            # rdmol = AddHs(rdmol)
            # logp and qed are the same irrespective of hydrogen addition

            logp = (Descriptors.MolLogP(rdmol))
            qed = QED.qed(rdmol)
            # props = QED.properties(rdmol)
            sascore = sascorer.calculateScore(rdmol)

            res = Image.new(pilimage.mode, (300, 350), (255,255,255))
            res.paste(pilimage, (0, 50))

            I1 = ImageDraw.Draw(res)

            params = ("logp = %.2f QED = %.2f SA = %.2f" % (logp, qed, sascore))
            font = ImageFont.truetype("myfont.ttf", 14)
            # print(params)
            # Add Text to an image
            I1.text((40, 25), params, fill=(0, 0, 0), font=font)

            # Display edited image
            res.save('./outputs_1/'+out_name+'/_'+str(i)+'.png')

            if(qed>mxqed):
                mxqed = qed
                mxi = i
                mxlogp = logp
                mxsa = sascore

        except Exception as e:
            print(e)
            print("Reconstruction failed")

    # print(ctx_lp.shape[0], "atoms")
    # num_lig_atoms = ctx_lp.shape[0]
    # comb_ind = (torch.combinations(torch.arange(0,num_lig_atoms,1), num_lig_atoms-2))
    # for ix,comb in enumerate(comb_ind):
    #     res = {}
    #     res['p'] = protein_ligand_pair['p']
    #     res['lp'] = ctx_lp[comb].numpy()
    #     res['lf'] = ctx_lf[comb].numpy()

    #     print(res['p'])
        
    #     try:
    #         reconstruct_from_struct(res['lp'], res['lf'], out_name+"/"+str(ix))
    #         torch.save(res, './outputs_1/'+out_name+'/'+str(ix)+'.pt')
    #     except:
    #         print("Reconstruction failed")
        
    if(mxi != 0):
        shutil.copy('./outputs_1/'+out_name+'/_'+str(mxi)+'.png', './outputs_1/mxQED/'+str(ix)+'.png')
        shutil.copy('./outputs_1/'+out_name+'/'+str(mxi)+'.sdf', './outputs_1/mxQED/'+str(ix)+'.sdf')
        shutil.rmtree('./outputs_1/'+out_name)

    if(log_vals): print("Done")

    p_path = protein_ligand_pair['p']
    l_path = protein_ligand_pair['l']
    
    return (logp, qed, sascore, mxqed, mxlogp, mxsa, mxi)

eps = 1000 # int(sys.argv[1]) 
log_vals = 0 # int(sys.argv[2])
print("Running",eps,"experiments")

logps = []
qeds = []
sascores = []
mxqeds = []
mxis = []
mxlogps = []
mxsas = []

pbar = tqdm(total=eps, position=0)
metrics = '\n'
for ep in range(eps):
    stats = tqdm(total=0, position=1, bar_format='{desc}')
    (logp, qed, sascore, mxqed, mxlogp, mxsa, mxi) = generate(ep)
    metrics += ("%d mxQED = %.2f mxlogp = %.2f mxSA = %.2f mxN =%d logp = %.2f QED = %.2f SA = %.2f %s %s\n" % (ep, mxqed, mxlogp, mxsa, mxi, logp, qed, sascore, p_path, l_path))
    if(logp != -10 and qed != -1 and sascore != -1):
        logps.append(logp)
        qeds.append(qed)
        sascores.append(sascore)
    if(mxqed != -1): 
        mxqeds.append(mxqed)
        mxis.append(mxi+3)
        mxlogps.append(mxlogp)
        mxsas.append(mxsa)

    pbar.update(1)
    if(log_vals): print('*'*60)
    
with open('./outputs_1/res.txt', 'w') as f:
    f.write(metrics)

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

plt.savefig('./logs/model_props.png')

plt.clf()

plt.figure(figsize=(20,20))

plt.subplot(3,1,1)
plt.xlabel('QED', labelpad=10)
sns.histplot(data=qeds, kde=True)

plt.subplot(3,1,2)
plt.xlabel('maxQED', labelpad=10)
sns.histplot(data=mxqeds, kde=True)

plt.subplot(3,1,3)
plt.xlabel('maxN', labelpad=10)
sns.histplot(data=mxis, kde=True)

plt.savefig('./logs/model_props_1.png')

plt.clf()

print(mxqeds, mxis)

plt.figure(figsize=(20,20))

plt.subplot(3,1,1)
plt.xlabel('maxlogp', labelpad=10)
sns.histplot(data=mxlogps, kde=True)

plt.subplot(3,1,2)
plt.xlabel('maxQED', labelpad=10)
sns.histplot(data=mxqeds, kde=True)

plt.subplot(3,1,3)
plt.xlabel('maxSA', labelpad=10)
sns.histplot(data=mxsas, kde=True)

plt.savefig('./logs/model_props_2.png')

# summary(actor)
# summary(critic)
        


