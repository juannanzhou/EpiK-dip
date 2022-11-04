import os
import numpy as np
import pandas as pd
import torch
import gpytorch
from .models import ExactGPModel

def set_data_path(path):
    global data_path
    data_path = path


def get_envs():
    geno_file_list = []
    for path, currentDirectory, files in os.walk(data_path + "96ghpptzvf-4/SData2/"):
        for file in files:
            if file.endswith("geno.txt"):
                geno_file_list.append(file)

    geno_file_list = list(set(geno_file_list))

    env_list = [file.split('_')[0] for file in geno_file_list]
    env_list = sorted(env_list)

    return env_list



def get_data(env):
    
    df = pd.read_csv(data_path + "96ghpptzvf-4/SData2/"+ env + "_geno.txt", sep='\t', nrows=5, engine='python')
    ids = list(df.columns[3:])

    geno_t = torch.load(data_path + env + '_matsui_geno_t.pt')
    geno_t = torch.tensor(geno_t, dtype=torch.float)
    geno_t = torch.transpose(geno_t, 0, 1)
    N, L = geno_t.shape

    pheno = pd.read_csv(data_path + "96ghpptzvf-4/SData6/" + env + "_pheno.txt", sep='\t', engine="python")
    pheno = pheno.set_index('geno')
    pheno = pheno.loc[ids]    
    
    return geno_t, pheno


def get_train_test(geno_t, pheno, sub, sub_t, output_device=0):
    train_x = geno_t[sub]
    train_y = torch.tensor(np.array(pheno.pheno[sub]), dtype=torch.float32)
    test_x = geno_t[sub_t]
    test_y = torch.tensor(np.array(pheno.pheno[sub_t]), dtype=torch.float32)
    train_x, train_y = train_x.contiguous(), train_y.contiguous()
    test_x, test_y = test_x.contiguous(), test_y.contiguous()
    train_x, train_y = train_x.to(output_device), train_y.to(output_device)
    test_x, test_y = test_x.to(output_device), test_y.to(output_device)
    return train_x, test_x, train_y, test_y


def train_model(model, 
                likelihood, 
                train_x, 
                train_y, 
                checkpoint_size, 
                preconditioner_size, 
                training_iter=300, 
                lr=.05):
    losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    model.train()
    
    for i in range(training_iter):
        if i%20 ==0:
            print(i)
            
        with gpytorch.beta_features.checkpoint_kernel(checkpoint_size):
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()  
            losses.append(loss.item())
            optimizer.step()        
            
            
            
# def train_model_cv(ker, train_x, train_y, val_x, val_y, training_iter, lr):
#     losses = []

#     """fitting hyperparameters of model by maximizing marginal log likelihood"""
#     # Use the adam optimizer, this includes GaussianLikelihood parameters
#     likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
#     model = ExactGPModel(train_x, train_y, likelihood, ker).to(output_device)    

#     optimizer = torch.optim.AdamW(model.parameters(), lr)

#     for i in range(training_iter):
#         if i%10 == 0:
#             print("working on iteration %f"%i)
#         # Zero gradients from previous iteration
#         optimizer.zero_grad()
#         # Output from model
#         model.eval()
#         f_preds = model(val_x).mean

#         # Calc loss and backprop gradients
#         loss = torch.norm(f_preds - val_y)
#         model.train()
#         loss.backward()
#         losses.append(loss.item())
#         optimizer.step()
#         del loss
#     del model

#     return ker, likelihood




def train_model_cv(ker, train_x, train_y, training_iter, lr, output_device=0):
    losses = []
    
    tr_size = np.min([20000, round(.5*train_x.shape[0])])
    val_size = np.min([10000, round(train_x.shape[0] - tr_size)])

    sub_tr = np.random.choice(range(len(train_x)), tr_size)
    sub_val = np.random.choice(list(set(range(len(train_x))).difference(sub_tr)), val_size)
    tr_x = train_x[sub_tr]
    tr_y = train_y[sub_tr]
    val_x = train_x[sub_val]
    val_y = train_y[sub_val]


    """fitting hyperparameters of model by maximizing marginal log likelihood"""
    # Use the adam optimizer, this includes GaussianLikelihood parameters
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
    model = ExactGPModel(tr_x, tr_y, likelihood, ker).to(output_device)    

    optimizer = torch.optim.AdamW(model.parameters(), lr)

    for i in range(training_iter):
        if i%10 == 0:
            print("working on iteration %i"%i)
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        model.eval()
        f_preds = model(val_x).mean

        # Calc loss and backprop gradients
        loss = torch.norm(f_preds - val_y)
        model.train()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        del loss
    del model

    return ker, likelihood
