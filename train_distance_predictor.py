import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from distance_predictor import DistancePredictor
from utils import load_file, find_nearest_protein_atoms_to_ligand, get_action_points # get_nearest_point

protein_z_id = {1:0, 6:1, 7:2, 8:3, 16:4, 34:5}
protein_z = {1,6,7,8,16,34}
max_aa = 20

ligand_z_id = {6:0,7:1,8:2,15:3,16:4,17:5}
ligand_z = [6,7,8,15,16,17]
NO_ATOM = len(ligand_z)

data_path = '/scratch/scratch1/hemanthramh/molgen/data/processed_2/'

dists = []
vals = []

index = load_file(data_path+'index.pickle')
sample_protein_ligand = load_file(data_path+index[0])

pfdim = len(sample_protein_ligand['pf'][0])
lfdim = len(sample_protein_ligand['lf'][0])

import torch
import torch.nn as nn

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def get_density(pos, lp):
    dist = torch.cdist(pos, lp)
    gauss = torch.max(torch.exp(-0.05*torch.pow(dist,2)), dim = 1).values
    # print(gauss.values)
    # print(gauss.shape, torch.min(dist), torch.max(gauss))
    return gauss

def get_least_distance(pos, lp):
    dist = torch.cdist(pos, lp)
    return torch.min(dist)

next_dists = []
max_scores = []
closest_dists = []

def run_experiment():
    
    file_name = data_path+index[np.random.randint(len(index))]
    # file_name = data_path+index[0]
    protein_ligand_pair = load_file(file_name)
    
    pp = torch.from_numpy(protein_ligand_pair['pp']).to(device)
    pf = torch.from_numpy(protein_ligand_pair['pf']).to(device)
    lp = torch.from_numpy(protein_ligand_pair['lp']).to(device)
    lf = torch.from_numpy(protein_ligand_pair['lf']).to(torch.float32).to(device)

    if(lp.shape[0] == 1): return False 

    rand_len = 1+np.random.randint(lp.shape[0]-1)
    # print(rand_len)
    # ctx_protein = find_nearest_protein_atoms_to_ligand(pp, lp, k = 100)
    ctx_pp = pp# [ctx_protein]
    ctx_pf = pf# [ctx_protein]
    ctx_lp = None if (rand_len == 0) else lp[:rand_len]
    ctx_lf = None if (rand_len == 0) else lf[:rand_len]

    rem_lp = lp[rand_len:]
    
    centers = torch.cat([ctx_pp]) if (rand_len == 0) else torch.cat([ctx_pp, ctx_lp])

    limits = [1.3, 1.8]
    resolution = 0.05

    next_pos = rem_lp[0]

    # all_actual_density = get_density(action_points, next_pos.view(1,3))
    # print(rand_len, action_points.shape)
    # sorted_actual_density_ix = torch.argsort(all_actual_density, descending=True)
    # action_points_sorted = action_points[sorted_actual_density_ix]
    
    # train_idx = torch.randperm(action_points.shape[0])[:1000]
    # train_idx = torch.cat([sorted_actual_density_ix[:4000],sorted_actual_density_ix[-1000:]])
    # action_points_train = action_points_sorted[::10]
    # print(all_actual_density[sorted_actual_density_ix])
    # print(all_actual_density[train_idx])
    
    # Model
    dists = model(
            protein_pos = ctx_pp,
            protein_atom_feature = ctx_pf,
            ligand_pos = ctx_lp,
            ligand_atom_feature = ctx_lf
        )

    actual_dists = torch.cdist(ctx_lp, next_pos.view(1,-1))

    loss_fn = nn.MSELoss()
    model_loss = loss_fn(dists, actual_dists)
    
    optimizer.zero_grad()
    model_loss.backward()
    optimizer.step()

    dists.detach_()

    # print(dists)
    # print(actual_dists)
    # print(model_loss)
    saved_losses.append(model_loss.detach().item())

    test = True
    if(test):
        action_points = get_action_points(ctx_lp, centers, limits, resolution)
        action_points_dist = torch.cdist(action_points, ctx_lp)
        action_points_dist_diff = torch.cdist(action_points_dist, dists.view(1,-1))
        min_err_ix = torch.argmin(action_points_dist_diff)
        pos_chosen = action_points[min_err_ix]
        actual_pos = next_pos
        dists = torch.cdist(pos_chosen.view(1,-1), rem_lp)
        score = torch.mean(torch.exp(-0.01*torch.pow(dists,2)), dim = 1)[0]
        dist = torch.norm(pos_chosen - next_pos)
        
        # print(action_points.shape)
        # print(action_points_dist.shape)
        # print(action_points_dist_diff.shape)
        # print(action_points_dist[min_err_ix])
        next_dists.append(dist.detach().item())
        max_scores.append(score.detach().item())
        closest_dists.append(torch.min(dists).detach().item())

    return True

if __name__ == "__main__":

    model = DistancePredictor(
        hidden_channels=128,
        protein_atom_feature_dim=pfdim,
        ligand_atom_feature_dim=lfdim
    )   
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 1e-4
        # betas=(0.90, 0.999)
    )

    try:
        checkpoint = torch.load('distance_predictor_checkpoint.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_losses = checkpoint['saved_losses']
        print('Distance Predictor Model Loaded')
    except:
        saved_losses = []
        print('No saved model found')

    eps = 5000  # int(sys.argv[1])
    print("Running",eps,"experiments")

    for ep in tqdm(range(eps)):
        run_experiment()

    torch.save({
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'saved_losses':saved_losses,
    }, 'distance_predictor_checkpoint.pt')
        # print('*************')

    # print(next_dists)
    # print(max_scores)
    print(closest_dists)

    sns.kdeplot(data=closest_dists)
    plt.xlabel('Distance to the closest ligand atom from the predicted point')
    plt.savefig('./logs/closest_dist.png')
    plt.clf()


    # torch.save({
    #     'dists':dists,
    #     'vals':vals,
    # }, 'density_predictor_report.pt')

    # summary(actor)
    # summary(critic)
