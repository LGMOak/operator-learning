import torch
import torch.nn as nn
import numpy as np

"""
DeepONet Operator Learning

Idea is to learn the mapping G: u0 -> u(x,t)
u0 is initial conditions and u(x,t) is the solution at any (x,t)

Branch net processes the initial condition function u0 which is samples from sensor points
Trunk net processes query coordinates (x,t)
Ouput inner product of branch and trunk outputs + bias
"""

class DeepONet(nn.Module):
    def __init__(self, branch_input_size=100, hidden_dim=128, p=128):
        """
        :param branch_input_size: number of sensor points for IC sampling
        :param hidden_dim: hidden layer width
        :param p: dimension of latent space (inner product dimension)
        """
        super().__init__()
        self.p = p

        # Branch network
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_size, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, p)
        )

        # trunk network
        self.trunk_net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, p), nn.GELU(),
        )

        # Learn the bias term
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0_sensors, query_points):
        """
        Operator learning is a one-many mapping so we need dimensions to match
        """
        batch_size = u0_sensors.shape[0]
        n_query = query_points.shape[1]

        # branch encodes the initial conditions
        branch_out = self.branch_net(u0_sensors)

        # Trunk encodes query coordinates
        # Reshape to handle queries at once
        query_flat = query_points.reshape(-1, 2)
        trunk_out = self.trunk_net(query_flat)
        trunk_out = trunk_out.reshape(batch_size, n_query, self.p)

        # Inner product for each point using Einstein notation shortcut
        output = torch.einsum('bp,bnp->bn', branch_out, trunk_out)
        output = output + self.bias

        return output

    def predict(self, u0_sensors, x_query, t_query):
        """
        Convenience method for prediction at specific (x,t) points
        Returns:
            output: (batch_size, n_t, n_x) - solution on grid
        """

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

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("DeepONet Architecture\n")

    model = DeepONet(
        branch_input_size=100,
        hidden_dim=128,
        p=128
    )

    print(f"Model created with {count_parameters(model):,} parameters")
    print("Branch Network:")
    print(model.branch_net)
    print("\nTrunk Network:")
    print(model.trunk_net)

    # Test forward pass
    batch_size = 16
    n_sensors = 100
    n_query = 1000

    u0_sensors = torch.randn(batch_size, n_sensors)
    query_points = torch.randn(batch_size, n_query, 2)

    output = model(u0_sensors, query_points)
    print(f"Initial conditions shape: {u0_sensors.shape}")
    print(f"Query points shape:       {query_points.shape}")
    print(f"Output shape:             {output.shape}")


    # Test prediction method
    x_query = np.linspace(-1, 1, 256)
    t_query = np.linspace(0, 1, 100)

    u0_test = torch.randn(4, n_sensors)
    output_grid = model.predict(u0_test, x_query, t_query)
    print(f"\nPrediction on grid:")
    print(f"Output grid shape: {output_grid.shape}")