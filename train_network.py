import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss
from tqdm import tqdm
import sys
import numpy as np
# from torchsummary import summary

from type_predictor import TypePredictor
from utils import load_file, find_nearest_protein_atoms_to_ligand#, get_action_points

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# critic = ValueFunction(
#     protein_atom_feature_dim=pfdim,
#     ligand_atom_feature_dim=lfdim,
#     hidden_channels=64,
#     k = 5
# )

actor = TypePredictor(
    num_classes=len(ligand_z),
    hidden_channels=64,
    protein_atom_feature_dim=pfdim,
    ligand_atom_feature_dim=lfdim,
    aggregate_k = 10
)

actor.to(device)

# density = DensityPredictor(
#     hidden_channels=32,
#     protein_atom_feature_dim=pfdim,
#     ligand_atom_feature_dim=lfdim
# )
# try:
#     density.load_state_dict(torch.load('density_predictor_checkpoint.pt')['model_state_dict'])
# except:
#     print("Saved Density Model Not Found")
#     exit()

params = list(actor.parameters())

optimizer = torch.optim.Adam(
    params,
    lr = 1e-4,
    # weight_decay=0,
    # betas=(0.90, 0.999)
)

try:
    checkpoint = torch.load('model_checkpoint.pt')
    actor.load_state_dict(checkpoint['actor_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    saved_type_losses = checkpoint['saved_type_losses']
    saved_terminate_losses = checkpoint['saved_terminate_losses']
    # saved_losses = checkpoint['saved_losses']
    print('Model Loaded')
except:
    saved_type_losses = []
    saved_terminate_losses = []
    # saved_losses = []
    print('No saved model found')

limits = [1, 1.5]
resolution = 0.2

stat = [0]*lfdim
conf_mat = np.zeros((len(ligand_z), len(ligand_z)), dtype=np.int64)

def run_experiment():
    
    file_name = data_path+index[np.random.randint(len(index))]
    protein_ligand_pair = load_file(file_name)
    n_ligand = len(protein_ligand_pair['lp'])
    
    randperm = torch.randperm(n_ligand)
    pp = torch.from_numpy(protein_ligand_pair['pp']).to(device)
    pf = torch.from_numpy(protein_ligand_pair['pf']).to(device)
    lp = torch.from_numpy(protein_ligand_pair['lp']).to(device)#[randperm]
    lf = torch.from_numpy(protein_ligand_pair['lf']).to(torch.float32).to(device)#[randperm]

    # Adding random ligand atoms
    rand_len = 0# np.random.randint(lp.shape[0])
    # print(rand_len)
    ctx_protein = find_nearest_protein_atoms_to_ligand(pp, lp, k = 75)
    ctx_pp = pp#[ctx_protein]
    ctx_pf = pf#[ctx_protein]
    ctx_lp = None if (rand_len == 0) else lp[:rand_len]
    ctx_lf = None if (rand_len == 0) else lf[:rand_len]
    
    rem_lp = lp[rand_len:]
    rem_lf = lf[rand_len:]
    
    # centers = torch.cat([ctx_pp]) if (rand_len == 0) else torch.cat([ctx_pp, ctx_lp])
    
    type_losses = []
    terminate_losses = []
    
    for i in range(rem_lf.shape[0]):

        # action_points = get_action_points(centers, limits, resolution)

        atom_cls, terminate = actor(
            pos_query = rem_lp[i].view(1,-1),
            protein_pos=ctx_pp, 
            protein_atom_feature=ctx_pf, 
            ligand_pos=ctx_lp, 
            ligand_atom_feature=ctx_lf
        )
        
        type_next = torch.argmax(rem_lf[i])
        pos_next = rem_lp[i]
        type_loss = -torch.log(atom_cls[type_next])
        terminate_loss = -torch.log(1-terminate)

        if(type_next == 0):
            if(np.random.random()<0.5): 
                type_losses.append(type_loss)
                stat[type_next] += 1
                conf_mat[type_next][torch.argmax(atom_cls)] += 1
        # elif(type_next == 2):
        #     if(np.random.random()<0.65): 
        #         type_losses.append(type_loss)
        #         stat[type_next] += 1
        else: 
            type_losses.append(type_loss)
            stat[type_next] += 1
            conf_mat[type_next][torch.argmax(atom_cls)] += 1
        terminate_losses.append(terminate_loss)
        
        new_atom_feature = torch.zeros((1, lfdim)).to(device)
        new_atom_feature[0,type_next] = 1

        if(ctx_lp is None):
            ctx_lp = torch.cat([pos_next.view(1,3)])
            ctx_lf = torch.cat([new_atom_feature])
        else:
            ctx_lp = torch.cat([ctx_lp, pos_next.view(1,3)])
            ctx_lf = torch.cat([ctx_lf, new_atom_feature])
            
        # centers = torch.cat([centers, pos_next.view(1,3)])

        if(log_vals):
            print("#        :", rand_len+i+1)
            print('Dist     :', atom_cls)
            print("Chosen   :", torch.argmax(atom_cls))
            print("Element  :", type_next)
            print("Position :", pos_next)
            print("Terminate:", terminate)
            print('-'*60)

    _, terminate = actor(
        pos_query = torch.zeros(1,3).to(device),
        protein_pos=ctx_pp, 
        protein_atom_feature=ctx_pf, 
        ligand_pos=ctx_lp, 
        ligand_atom_feature=ctx_lf
    )
    _.detach()
    terminate_loss = -torch.log(terminate)
    terminate_losses.append(terminate_loss)

    optimizer.zero_grad()

    terminate_losses = torch.stack(terminate_losses)
    terminator_loss = (terminate_losses).mean()
    saved_terminate_losses.append(terminator_loss.detach().item())
    terminator_loss.backward()

    if(len(type_losses) > 0):
        type_losses = torch.stack(type_losses)
        type_predictor_loss = (type_losses).mean()
        saved_type_losses.append(type_predictor_loss.detach().item())
        type_predictor_loss.backward()

    optimizer.step()

    # saved_losses.append(loss.detach().item()/(i+1))
    
    if(log_vals): print("Step Done")
    return True

eps = 90000 # int(sys.argv[1]) 
log_vals = 0 # int(sys.argv[2])
print("Running",eps,"experiments")

pbar = tqdm(total=eps, position=0)
stats = tqdm(total=0, position=1, bar_format='{desc}')
for ep in range(eps):
    run_experiment()

    stats.set_description_str('Stat:'+str(stat))
    pbar.update(1)

    if(log_vals): print('*'*60)

torch.save({
    'actor_state_dict':actor.state_dict(),
    'optimizer_state_dict':optimizer.state_dict(),
    'saved_type_losses':saved_type_losses,
    'saved_terminate_losses':saved_terminate_losses,
    # 'saved_losses':saved_losses,
}, 'model_checkpoint.pt')

print("Confusion Matrix\n", conf_mat)
# summary(actor)
# summary(critic)
        


