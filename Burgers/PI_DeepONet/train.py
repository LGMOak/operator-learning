import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pi_deeponet import PIDeepONet, compute_pde_residual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
BATCH_SIZE = 32
EPOCHS = 5000
LEARNING_RATE = 1e-3

# Loss weights
LAMBDA_IC = 10.0  # Initial condition loss
LAMBDA_PDE = 1.0  # PDE residual loss
LAMBDA_DATA = 10.0  # Solution data loss

NU = 0.01 / np.pi

N_COLLOCATION = 4000
N_DATA = 1000  # Data points per batch

DATA_PATH = '../data/burgers_data_train.pkl'
MODEL_SAVE_PATH = './models/pi_deeponet_burgers.pth'


def load_data(path):
    """Load full dataset including solutions"""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    u0 = data['initial_conditions']
    u_sol = data['solutions']  # NEW: Load solutions
    x = data['x']
    t = data['t']  # NEW: Load time coordinates

    return (
        torch.tensor(u0, dtype=torch.float32),
        u_sol,  # Keep as numpy for indexing
        torch.tensor(x, dtype=torch.float32),
        t  # Keep as numpy for indexing
    )


def train():
    print(f"Training PI-DeepONet on {device}")
    print(f"Loss weights: λ_IC={LAMBDA_IC}, λ_PDE={LAMBDA_PDE}, λ_DATA={LAMBDA_DATA}\n")

    print("Loading data...")
    u0_train, u_sol_train, x_coords, t_coords = load_data(DATA_PATH)
    n_train = len(u0_train)
    nx_res = u0_train.shape[1]
    nt_res = len(t_coords)

    print(f"Loaded {n_train} initial conditions")
    print(f"Spatial resolution: {nx_res}")
    print(f"Temporal resolution: {nt_res}\n")

    model = PIDeepONet(branch_input_size=nx_res, hidden_dim=128, p=128, num_layers=7).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=500, verbose=True
    )

    x_coords = x_coords.to(device)

    loss_history = []
    ic_loss_history = []
    pde_loss_history = []
    data_loss_history = []

    for epoch in range(EPOCHS):
        model.train()

        # Sample random ICs
        indices = torch.randint(0, n_train, (BATCH_SIZE,))
        batch_u0 = u0_train[indices].to(device)

        optimiser.zero_grad()

        # IC Loss
        t_zeros = torch.zeros(nx_res, device=device)
        xt_ic = torch.stack([x_coords, t_zeros], dim=1)  # (nx, 2)
        xt_ic = xt_ic.unsqueeze(0).expand(BATCH_SIZE, -1, -1)  # (B, nx, 2)

        u_pred_ic = model(batch_u0, xt_ic)
        loss_ic = torch.mean((u_pred_ic - batch_u0) ** 2)

        # PDE Loss
        collocation = torch.rand(BATCH_SIZE, N_COLLOCATION, 2, device=device)
        collocation[:, :, 0] = collocation[:, :, 0] * 2 - 1  # x
        collocation[:, :, 1] = collocation[:, :, 1]  # t
        collocation.requires_grad_(True)

        residual = compute_pde_residual(model, batch_u0, collocation, nu=NU)
        loss_pde = torch.mean(residual ** 2)

        # Vectorised Data loss

        # Select random points
        t_idx = np.random.choice(nt_res, N_DATA, replace=True)
        x_idx = np.random.choice(nx_res, N_DATA, replace=True)

        # Gather x and t values
        x_val = x_coords[x_idx].cpu().numpy()
        t_val = t_coords[t_idx]

        # Stack into (N_DATA, 2)
        xt_single = np.stack([x_val, t_val], axis=1)
        xt_single = torch.tensor(xt_single, dtype=torch.float32, device=device)

        # Expand for batch: (B, N_DATA, 2)
        xt_data = xt_single.unsqueeze(0).expand(BATCH_SIZE, -1, -1)

        # Prepare targets (True Solution)
        # Use vectorised indexing: sol[batch_indices, t_indices, x_indices]
        batch_indices_np = indices.cpu().numpy()

        # Shape: (BATCH_SIZE, N_DATA)
        u_true_batch = u_sol_train[batch_indices_np[:, None], t_idx[None, :], x_idx[None, :]]
        u_true_data = torch.tensor(u_true_batch, dtype=torch.float32, device=device)

        # Predict
        u_pred_data = model(batch_u0, xt_data)

        # Loss
        loss_data = torch.mean((u_pred_data - u_true_data) ** 2)

        # Total loss
        loss = LAMBDA_IC * loss_ic + LAMBDA_PDE * loss_pde + LAMBDA_DATA * loss_data

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        loss_history.append(loss.item())
        ic_loss_history.append(loss_ic.item())
        pde_loss_history.append(loss_pde.item())
        data_loss_history.append(loss_data.item())

        scheduler.step(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss.item():.6f} | "
                  f"IC: {loss_ic.item():.6f} | PDE: {loss_pde.item():.6f} | "
                  f"Data: {loss_data.item():.6f}")

    print("\nTraining phase 2: L-BFGS optimisation")

    lbfgs_batch = u0_train.to(device)
    n_full = len(lbfgs_batch)
    N_LBFGS_COLLOC = 100
    N_LBFGS_DATA = 200

    # Fixed collocation points
    fixed_colloc = torch.rand(n_full, N_LBFGS_COLLOC, 2, device=device)
    fixed_colloc[:, :, 0] = fixed_colloc[:, :, 0] * 2 - 1
    fixed_colloc[:, :, 1] = fixed_colloc[:, :, 1]
    fixed_colloc.requires_grad_(True)

    # Sample data points once for L-BFGS
    t_idx_lbfgs = np.random.choice(nt_res, N_LBFGS_DATA, replace=True)
    x_idx_lbfgs = np.random.choice(nx_res, N_LBFGS_DATA, replace=True)

    xt_data_lbfgs = []
    u_true_data_lbfgs = []

    for batch_idx in range(n_full):
        x_samples = x_coords[x_idx_lbfgs]
        t_samples = torch.tensor(t_coords[t_idx_lbfgs], dtype=torch.float32, device=device)
        xt_sample = torch.stack([x_samples, t_samples], dim=1)

        u_true = torch.tensor(
            [u_sol_train[batch_idx, ti, xi] for ti, xi in zip(t_idx_lbfgs, x_idx_lbfgs)],
            dtype=torch.float32,
            device=device
        )

        xt_data_lbfgs.append(xt_sample)
        u_true_data_lbfgs.append(u_true)

    xt_data_lbfgs = torch.stack(xt_data_lbfgs)
    u_true_data_lbfgs = torch.stack(u_true_data_lbfgs)

    optimiser_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimiser_lbfgs.zero_grad()

        # IC loss
        t_zeros = torch.zeros(nx_res, device=device)
        xt_ic = torch.stack([x_coords, t_zeros], dim=1)
        xt_ic = xt_ic.unsqueeze(0).expand(n_full, -1, -1)

        u_pred_ic = model(lbfgs_batch, xt_ic)
        loss_ic = torch.mean((u_pred_ic - lbfgs_batch) ** 2)

        # PDE Loss
        res = compute_pde_residual(model, lbfgs_batch, fixed_colloc, nu=NU)
        loss_pde = torch.mean(res ** 2)

        # Data Loss (NEW!)
        u_pred_data = model(lbfgs_batch, xt_data_lbfgs)
        loss_data = torch.mean((u_pred_data - u_true_data_lbfgs) ** 2)

        # Total Loss
        loss = LAMBDA_IC * loss_ic + LAMBDA_PDE * loss_pde + LAMBDA_DATA * loss_data
        loss.backward()

        # Store values
        closure.loss_val = loss.item()
        closure.ic_val = loss_ic.item()
        closure.pde_val = loss_pde.item()
        closure.data_val = loss_data.item()

        loss_history.append(closure.loss_val)
        ic_loss_history.append(closure.ic_val)
        pde_loss_history.append(closure.pde_val)
        data_loss_history.append(closure.data_val)

        return loss

    lbfgs_epochs = 100

    for i in range(lbfgs_epochs):
        optimiser_lbfgs.step(closure)

        print(f"L-BFGS Step {i + 1}/{lbfgs_epochs} | "
              f"Loss: {closure.loss_val:.6f} | "
              f"IC: {closure.ic_val:.6f} | "
              f"PDE: {closure.pde_val:.6f} | "
              f"Data: {closure.data_val:.6f}")

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # Plot training loss
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 2)
    plt.plot(ic_loss_history)
    plt.yscale('log')
    plt.title('IC Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 3)
    plt.plot(pde_loss_history)
    plt.yscale('log')
    plt.title('PDE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 4)
    plt.plot(data_loss_history)
    plt.yscale('log')
    plt.title('Data Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./figures/pi_deeponet_training_loss.png')
    print("Training loss plot saved to ./figures/pi_deeponet_training_loss.png")

    return model


if __name__ == "__main__":
    train()