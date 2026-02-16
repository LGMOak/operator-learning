import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from navier_stokes import Navier_Stokes

MODEL_PATH = './models/pi_deeponet_ns.pth'
DATA_PATH = './data/ns_data.pkl'
FIGURE_DIR = './figures'
os.makedirs(FIGURE_DIR, exist_ok=True)

TARGET_RE_LIST = [150, 300, 500, 750, 950]
MAX_ITER = 50
LR = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    model = Navier_Stokes(hidden_dim=128, p=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    for param in model.parameters(): 
        param.requires_grad = False
    model.eval()
    
    return data, model

def get_smart_sensors(data, target_re, n_sensors):
    """Extracts interior sensors to avoid dead boundary zones."""
    Re_all = data['Re']
    idx = (np.abs(Re_all - target_re)).argmin()
    actual_re = Re_all[idx]
    
    psi_true = data['psi'][idx]
    w_true = data['w'][idx]
    X, Y = np.meshgrid(data['x'], data['y'])
    
    ny, nx = psi_true.shape
    interior_indices = [(i, j) for i in range(2, ny-2) for j in range(2, nx-2)]
    
    selected_indices = np.random.choice(len(interior_indices), n_sensors, replace=False)
    
    sensor_coords, sensor_vals_psi, sensor_vals_w = [], [], []
    for k in selected_indices:
        i, j = interior_indices[k]
        sensor_coords.append([X[i, j], Y[i, j]])
        sensor_vals_psi.append(psi_true[i, j])
        sensor_vals_w.append(w_true[i, j])
        
    sensor_xy = torch.tensor(np.array(sensor_coords), dtype=torch.float32).to(device).unsqueeze(0)
    sensor_psi = torch.tensor(np.array(sensor_vals_psi), dtype=torch.float32).to(device)
    sensor_w = torch.tensor(np.array(sensor_vals_w), dtype=torch.float32).to(device)
    
    return actual_re, sensor_xy, sensor_psi, sensor_w

def inverse_discovery():
    initial_guess = 500.0
    n_sensors = 1000
    print(f"Initial Guess: {initial_guess} | Sensors: {n_sensors}\n")
    
    data, model = load_data()
    histories, actual_res = {}, {}

    for target_re in TARGET_RE_LIST:
        actual_re, sensor_xy, sensor_psi, sensor_w = get_smart_sensors(data, target_re, n_sensors)
        actual_res[target_re] = actual_re
        
        re_param = torch.nn.Parameter(torch.tensor([[initial_guess]], dtype=torch.float32).to(device))
        optimiser = torch.optim.LBFGS([re_param], lr=LR, max_iter=MAX_ITER, 
                                      history_size=100, line_search_fn='strong_wolfe')
        
        eval_history = [initial_guess] 
        
        def closure():
            optimiser.zero_grad()
                
            re_norm = re_param / 1000.0
            preds = model(re_norm, sensor_xy)
            
            loss = torch.mean((preds[0, :, 0] - sensor_psi)**2) + \
                torch.mean((preds[0, :, 1] - sensor_w)**2)
            loss.backward()
            
            eval_history.append(re_param.item())
            return loss

        optimiser.step(closure)
        
        final_re = re_param.item()
        error = abs(final_re - actual_re) / actual_re * 100
        print(f"Target: {target_re:3d} | Found: {final_re:.2f} (Error: {error:.2f}%)")
        histories[target_re] = eval_history

    print(f"\nGenerating convergence plots...")
    fig, axes = plt.subplots(1, len(TARGET_RE_LIST), figsize=(4 * len(TARGET_RE_LIST), 4), sharey=True)
    
    for ax, target_re in zip(axes, TARGET_RE_LIST):
        hist = histories[target_re]
        true_val = actual_res[target_re]
        
        ax.plot(hist, 'b.-', linewidth=2, markersize=8, label='Estimate')
        ax.axhline(true_val, color='r', linestyle='--', linewidth=2, label=f'True={true_val:.1f}')
        ax.axhline(initial_guess, color='g', linestyle=':', label=f'Init={initial_guess}')
        
        ax.set_title(f'Target Re = {target_re}')
        ax.set_xlabel('Function Evaluations')
        if ax == axes[0]:
            ax.set_ylabel('Reynolds Number')
            
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
    plt.tight_layout()
    save_path = f'{FIGURE_DIR}/inverse_convergence.png'
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

def inverse_discovery_sparse():
    print("Sparse sensor discovery (100 Sensors, Multi-Start)")
    GUESSES = [200.0, 500.0, 800.0]
    N_SENSORS = 50
    
    data, model = load_data()
    histories = {re: {} for re in TARGET_RE_LIST}
    actual_res = {}

    for target_re in TARGET_RE_LIST:
        actual_re, sensor_xy, sensor_psi, sensor_w = get_smart_sensors(data, target_re, N_SENSORS)
        actual_res[target_re] = actual_re
        
        print(f"Target: {target_re:3d}")
        
        for guess in GUESSES:
            re_param = torch.nn.Parameter(torch.tensor([[guess]], dtype=torch.float32).to(device))
            optimiser = torch.optim.LBFGS([re_param], lr=LR, max_iter=MAX_ITER, history_size=100, line_search_fn='strong_wolfe')
            
            eval_history = [guess] 
            
            def closure():
                optimiser.zero_grad()
                re_norm = re_param / 1000.0 
                preds = model(re_norm, sensor_xy)
                loss = torch.mean((preds[0, :, 0] - sensor_psi)**2) + torch.mean((preds[0, :, 1] - sensor_w)**2)
                loss.backward()
                eval_history.append(re_param.item())
                return loss

            optimiser.step(closure)
                
            final_re = re_param.item()
            error = abs(final_re - actual_re) / actual_re * 100
            print(f"  Guess {guess}: Found {final_re:.2f} (Error: {error:.2f}%)")
            histories[target_re][guess] = eval_history

    colors = {200.0: 'blue', 500.0: 'orange', 800.0: 'green'}
    fig, axes = plt.subplots(1, len(TARGET_RE_LIST), figsize=(15, 4), sharey=True)
    
    for ax, target_re in zip(axes, TARGET_RE_LIST):
        for guess in GUESSES:
            ax.plot(histories[target_re][guess], '.-', color=colors[guess], alpha=0.7, label=f'Init: {guess}')
            
        ax.axhline(actual_res[target_re], color='r', linestyle='--', label=f'True={actual_res[target_re]:.0f}')
        ax.set_title(f'Target: {target_re}')
        ax.set_xlabel('Evaluations')
        if ax == axes[0]: ax.set_ylabel('Reynolds Number')
        ax.grid(True, alpha=0.3)
        if ax == axes[-1]: ax.legend(loc='best', fontsize=8)
        
    plt.suptitle('Sparse Sensor Optimisation')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/inverse_sparse_convergence.png', dpi=150)
    print("Saved sparse plot.\n")

if __name__ == "__main__":
    inverse_discovery()
    inverse_discovery_sparse()