import torch
import torch.nn as nn
import numpy as np


class ModifiedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=7):
        super().__init__()
        self.phi = nn.Tanh()
        self.u_gate = nn.Linear(input_dim, hidden_dim)
        self.v_gate = nn.Linear(input_dim, hidden_dim)
        self.h_init = nn.Linear(input_dim, hidden_dim)

        # Hidden layers (Z gates)
        self.z_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.last_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        u = self.phi(self.u_gate(x))
        v = self.phi(self.v_gate(x))
        h = self.phi(self.h_init(x))
        for z_layer in self.z_layers:
            z = self.phi(z_layer(h))
            h = (1 - z) * u + z * v
        return self.last_layer(h)

class FourierFeatureEmbedding(nn.Module):
    def __init__(self, in_features=2, out_features=64, scale=5.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Fixed Gaussian weights
        self.B = nn.Parameter(torch.randn(in_features, out_features) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B

        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PIDeepONet(nn.Module):
    def __init__(self, branch_input_size=100, hidden_dim=128, p=128, num_layers=7):
        super().__init__()
        self.p = p
        # Branch processes initial conditions (u0)
        self.branch_net = ModifiedMLP(branch_input_size, hidden_dim, p, num_layers)

        # Fourier Embedding
        self.fourier_dim = 128
        self.embedding = FourierFeatureEmbedding(in_features=2, out_features=self.fourier_dim//2, scale=2.0)

        # Trunk processes coordinates (x, t)
        self.trunk_net = ModifiedMLP(self.fourier_dim, hidden_dim, p, num_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0, xt):
        B = u0.shape[0]
        N = xt.shape[1]

        branch_out = self.branch_net(u0)  # (B, p)

        xt_flat = xt.reshape(-1, 2)
        xt_embedded = self.embedding(xt_flat)

        trunk_out = self.trunk_net(xt_embedded)  # (B*N, p)
        trunk_out = trunk_out.reshape(B, N, self.p)
        return torch.einsum('bp,bnp->bn', branch_out, trunk_out) + self.bias

    def predict(self, u0_sensors, x_query, t_query):
        device = next(self.parameters()).device
        batch_size = u0_sensors.shape[0]

        # Create meshgrid
        T, X = torch.meshgrid(torch.tensor(t_query, dtype=torch.float32),
                              torch.tensor(x_query, dtype=torch.float32),
                              indexing='ij')

        # Flatten and stack
        xt = torch.stack([X.flatten(), T.flatten()], dim=1).to(device)  # (n_t*n_x, 2)

        # Expand for batch
        xt_batch = xt.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_t*n_x, 2)

        # prediction
        with torch.no_grad():
            output = self.forward(u0_sensors, xt_batch)  # (batch, n_t*n_x)

        # Reshape to grid (batch, time, space)
        output = output.reshape(batch_size, len(t_query), len(x_query))

        return output

def compute_pde_residual(model, u0, collocation_points, nu=0.01/np.pi):
    """
    Computes Burgers PDE Residual: u_t + u*u_x - nu*u_xx =0
    """
    # Forward pass
    u = model(u0, collocation_points)

    u_flat = u.reshape(-1)

    # first derivatives
    grads = torch.autograd.grad(u_flat, collocation_points, grad_outputs=torch.ones_like(u_flat),
                                create_graph=True, retain_graph=True)[0]

    u_x = grads[:, :, 0]
    u_t = grads[:, :, 1]

    u_x_flat = u_x.reshape(-1)
    u_xx = torch.autograd.grad(u_x_flat, collocation_points, grad_outputs=torch.ones_like(u_x_flat),
                               create_graph=True, retain_graph=True)[0][:, :, 0]

    residual = u_t + u * u_x - nu * u_xx

    return residual

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("PI-DeepONet Architecture\n")

    model = PIDeepONet(
        branch_input_size=100,
        hidden_dim=128,
        p=128
    )

    print(f"Model created with {count_parameters(model):,} parameters")
    print("Branch Network (Tanh):")
    print(model.branch_net)
    print("\nTrunk Network (Tanh):")
    print(model.trunk_net)

    # Test forward pass
    batch_size = 16
    n_sensors = 100
    n_query = 1000

    u0_sensors = torch.randn(batch_size, n_sensors)
    query_points = torch.randn(batch_size, n_query, 2)

    output = model(u0_sensors, query_points)
    print(f"\nInitial conditions shape: {u0_sensors.shape}")
    print(f"Query points shape:       {query_points.shape}")
    print(f"Output shape:             {output.shape}")

    # Test PDE residual
    print("\nTesting PDE residual computation...")
    query_points.requires_grad_(True)
    residual = compute_pde_residual(model, u0_sensors, query_points)
    print(f"Residual shape: {residual.shape}")
    print(f"Mean residual:  {residual.mean().item():.6f}")

