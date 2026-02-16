import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from navier_stokes import Navier_Stokes

# Configuration
MODEL_PATH = './models/pi_deeponet_ns.pth'
DATA_PATH = './data/ns_data.pkl'
FIGURE_DIR = './figures/evaluation'
os.makedirs(FIGURE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_velocity_from_psi(psi, dx, dy):
    """
    Compute u = dpsi/dy and v = -dpsi/dx using central differences
    psi: (Ny, Nx) numpy array
    """
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    
    # Central difference for interior
    u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dy)
    v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2 * dx)
    
    # Top lid boundary condition
    u[-1, :] = 1.0
    
    return u, v

def evaluate():
    print(f"Loading model from {MODEL_PATH}")
    
    # 1. Load Data
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
        
    # Convert lists to arrays if not already
    Re_list = data['Re']
    psi_truth = data['psi']
    x = data['x']
    y = data['y']
    
    # Grid setup for prediction
    X, Y = np.meshgrid(x, y)
    nx, ny = len(x), len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Prepare coordinates tensor for the trunk net
    grid_tensor = torch.tensor(np.stack([X, Y], axis=-1).reshape(-1, 2), dtype=torch.float32).to(device)
    
    # 2. Load Model
    model = Navier_Stokes(hidden_dim=128, p=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 3. Select Test Cases (Low, Mid, High Re)
    # Find indices closest to specific Re values
    target_Re = [100, 400, 1000]
    test_indices = []
    for re_val in target_Re:
        idx = (np.abs(Re_list - re_val)).argmin()
        test_indices.append(idx)
        
    # 4. Loop through test cases and visualize
    for i, idx in enumerate(test_indices):
        re_val = Re_list[idx]
        print(f"Evaluating Re = {re_val:.1f}...")
        
        # Ground Truth
        psi_gt = psi_truth[idx]
        u_gt, v_gt = compute_velocity_from_psi(psi_gt, dx, dy)
        vel_mag_gt = np.sqrt(u_gt**2 + v_gt**2)
        
        # Prediction
        # Prepare Branch Input (Reynolds Number)
        re_norm_val = re_val / 1000.0
        re_tensor = torch.tensor([[re_norm_val]], dtype=torch.float32).to(device)
        
        # Repeat Re for batch size 1 to match Trunk dimensions implies broadcasting in model
        # But our model expects (Batch, 1) and (Batch, N, 2)
        # We will treat the entire grid as one "sample" in the batch

        
        with torch.no_grad():
            # Expand Re to batch size 1
            # Expand Grid to (1, N_points, 2)
            grid_batch = grid_tensor.unsqueeze(0)
            
            # Forward Pass
            preds = model(re_tensor, grid_batch) # Output: (1, N_points, 2)
            
            # Extract Psi (Index 0)
            psi_pred_flat = preds[0, :, 0].cpu().numpy()
            
            # Reshape back to grid (Ny, Nx) -> Be careful with meshgrid indexing!
            # Meshgrid 'xy' vs 'ij'. If data used 'ij', we use 'ij'.
            # Based on generate_data.py: np.meshgrid(x, y) defaults to 'xy' (Cartesian). 
            # BUT the solver often treats indices as (row, col) which is (y, x).
            # Let's assume standard matrix indexing (Ny, Nx).
            psi_pred = psi_pred_flat.reshape(ny, nx)
            
        # Compute Derived Variables
        u_pred, v_pred = compute_velocity_from_psi(psi_pred, dx, dy)
        vel_mag_pred = np.sqrt(u_pred**2 + v_pred**2)
        
        # Error
        psi_error = np.abs(psi_gt - psi_pred)
        vel_error = np.abs(vel_mag_gt - vel_mag_pred)
        
        # --- PLOTTING ---
        plot_comparison(X, Y, 
                       psi_gt, psi_pred, psi_error,
                       vel_mag_gt, vel_mag_pred, vel_error,
                       u_pred, v_pred,
                       re_val, i)

def plot_comparison(X, Y, 
                   psi_gt, psi_pred, psi_err, 
                   vel_gt, vel_pred, vel_err, 
                   u_pred, v_pred,
                   re, idx):
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plt.suptitle(f'Lid Driven Cavity Analysis (Re={re:.1f})', fontsize=16)
    
    # --- Row 1: Streamfunction ---
    
    # Truth
    ax = axes[0, 0]
    cf = ax.contourf(X, Y, psi_gt, levels=50, cmap='viridis')
    plt.colorbar(cf, ax=ax)
    ax.set_title('Streamfunction (Truth)')
    ax.set_aspect('equal')
    
    # Pred (with Streamlines)
    ax = axes[0, 1]
    cf = ax.contourf(X, Y, psi_pred, levels=50, cmap='viridis')
    # Add streamlines to visualize the vortex
    ax.streamplot(X, Y, u_pred, v_pred, color='white', linewidth=0.5, density=0.7)
    plt.colorbar(cf, ax=ax)
    ax.set_title('Streamfunction (Pred) + Streamlines')
    ax.set_aspect('equal')
    
    # Error
    ax = axes[0, 2]
    cf = ax.contourf(X, Y, psi_err, levels=50, cmap='magma')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f'Psi Error (Max: {psi_err.max():.2e})')
    ax.set_aspect('equal')
    
    # --- Row 2: Velocity Magnitude ---
    
    # Truth
    ax = axes[1, 0]
    cf = ax.contourf(X, Y, vel_gt, levels=50, cmap='plasma')
    plt.colorbar(cf, ax=ax)
    ax.set_title('Velocity Mag (Truth)')
    ax.set_aspect('equal')
    
    # Pred
    ax = axes[1, 1]
    cf = ax.contourf(X, Y, vel_pred, levels=50, cmap='plasma')
    plt.colorbar(cf, ax=ax)
    ax.set_title('Velocity Mag (Pred)')
    ax.set_aspect('equal')
    
    # Error
    ax = axes[1, 2]
    cf = ax.contourf(X, Y, vel_err, levels=50, cmap='magma')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f'Velocity Error (Max: {vel_err.max():.2e})')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'./figures/evaluation/re_{int(re)}.png', dpi=150)
    plt.close()
    print(f"Saved figure for Re={re}")

if __name__ == "__main__":
    evaluate()