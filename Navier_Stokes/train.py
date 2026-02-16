from navier_stokes import Navier_Stokes, compute_residual
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 16          # Lower batch size due to 2D memory cost
EPOCHS = 10000
BURN_IN = 2000           # Only train on data for first 2000 epochs
LEARNING_RATE = 5e-4     # lower LR for stability

LAMBDA_DATA = 20.0
LAMBDA_POISSON = 1.0
LAMBDA_VORTICITY = 1.0
LAMBDA_BC = 5.0

N_COLLOCATION = 2000 
N_DATA_POINTS = 500

DATA_PATH = './data/ns_data.pkl'
MODEL_SAVE_PATH = './models/pi_deeponet_ns.pth'
os.makedirs('./models', exist_ok=True)
os.makedirs('./figures', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(path):
    print(f"Loading data from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    Re_list = torch.tensor(data['Re'], dtype=torch.float32).unsqueeze(1)
    
    # train against psi and omega 
    psi = torch.tensor(data['psi'], dtype=torch.float32) # (N_samples, Nx, Ny)
    w = torch.tensor(data['w'], dtype=torch.float32)     # (N_samples, Nx, Ny)

    x = torch.tensor(data['x'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.float32)    

    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    # flatten to (N_points, 2) after stacking
    grid_coords = torch.stack([X, Y], dim=-1).reshape(-1, 2) 
    
    return Re_list, psi, w, grid_coords

def train():
    print(f"Using device: {device}")
    
    Re_train, psi_train, w_train, grid_coords = load_data(DATA_PATH)
    n_samples = len(Re_train)
    
    Re_train = Re_train.to(device)
    psi_train = psi_train.to(device)
    w_train = w_train.to(device)
    grid_coords = grid_coords.to(device)
    
    model = Navier_Stokes(hidden_dim=128, p=128).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=1000, verbose=True)
    
    history = {'loss': [], 'data': [], 'poisson': [], 'vorticity': []}
    
    for epoch in range(EPOCHS):
        model.train()
        optimiser.zero_grad()
        
        # Data loss
        idx_batch = torch.randint(0, n_samples, (BATCH_SIZE,))
        batch_Re = Re_train[idx_batch]
        
        idx_coords = torch.randint(0, grid_coords.shape[0], (N_DATA_POINTS,))
        batch_xy = grid_coords[idx_coords].unsqueeze(0).expand(BATCH_SIZE, -1, -1)
        
        batch_psi_exact = psi_train[idx_batch].reshape(BATCH_SIZE, -1)[:, idx_coords]
        batch_w_exact = w_train[idx_batch].reshape(BATCH_SIZE, -1)[:, idx_coords]

        # normalise Reynolds
        batch_Re_norm = batch_Re / 1000.0
        
        # Forward pass
        preds = model(batch_Re_norm, batch_xy) # (B, N, 2)
        psi_pred = preds[:, :, 0]
        w_pred = preds[:, :, 1]
        
        loss_data_psi = torch.mean((psi_pred - batch_psi_exact)**2)
        loss_data_w = torch.mean((w_pred - batch_w_exact)**2)
        loss_data = loss_data_psi + loss_data_w
        
        # Physics Loss
        if epoch < BURN_IN:
            # ignore physics
            w_poisson = 0.0
            w_vorticity = 0.0
            loss_poisson = torch.tensor(0.0).to(device)
            loss_vorticity = torch.tensor(0.0).to(device)
        else:
            w_poisson = LAMBDA_POISSON
            w_vorticity = LAMBDA_VORTICITY

            Re_phys = torch.rand(BATCH_SIZE, 1, device=device) * 900 + 100
            coord_phys = torch.rand(BATCH_SIZE, N_COLLOCATION, 2, device=device)

            res_poisson, res_vorticity = compute_residual(model, Re_phys, coord_phys)
            loss_poisson = torch.mean(res_poisson ** 2)
            loss_vorticity = torch.mean(res_vorticity ** 2)

        # Total Loss
        loss = (LAMBDA_DATA * loss_data +
                w_poisson * loss_poisson +
                w_vorticity * loss_vorticity)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step(loss)
        
        curr_loss = loss.item()
        history['loss'].append(curr_loss)
        history['data'].append(loss_data.item())
        history['poisson'].append(loss_poisson.item())
        history['vorticity'].append(loss_vorticity.item())
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}/{EPOCHS} "
                f"Loss: {curr_loss:.4f} "
                f"Data: {loss_data.item():.4f} "
                f"Poisson: {loss_poisson.item():.4f} "
                f"Vorticity: {loss_vorticity.item():.4f} ")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    plot_losses(history)
        
def plot_losses(history):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(history['loss'])
    plt.yscale('log')
    plt.title('Total Loss')
    
    plt.subplot(1, 4, 2)
    plt.plot(history['data'])
    plt.yscale('log')
    plt.title('Data Loss')

    plt.subplot(1, 4, 3)
    plt.plot(history['poisson'])
    plt.yscale('log')
    plt.title('Poisson Residual')
    
    plt.subplot(1, 4, 4)
    plt.plot(history['vorticity'])
    plt.yscale('log')
    plt.title('Vorticity Residual')
    
    plt.tight_layout()
    plt.savefig('./figures/ns_training_loss.png')
    print("Loss plot saved.")

if __name__ == "__main__":
    train()
        
        
    
    
    
    