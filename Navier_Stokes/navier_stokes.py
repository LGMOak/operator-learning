import numpy as np
import torch
import torch.nn as nn

class ModifiedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super().__init__()
        self.phi = nn.GELU()
        self.u_gate = nn.Linear(input_dim, hidden_dim)
        self.v_gate = nn.Linear(input_dim, hidden_dim)
        self.h_init = nn.Linear(input_dim, hidden_dim)

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
    
class Navier_Stokes(nn.Module):
    def __init__(self, hidden_dim=128, p=128):
        super().__init__()
        self.p =p
        
        # BranchNet takes Reynolds number as input
        self.branch_net = ModifiedMLP(1, hidden_dim, p * 2, num_layers=5)
        
        # TrunkNet takes coordinates (x,y)
        self.trunk_net = ModifiedMLP(2, hidden_dim, p * 2, num_layers=5)
        
        # two bias terms for each variable psi and w
        self.bias = nn.Parameter(torch.zeros(2))
        
    def forward(self, re_branch, xy_trunk):
        B = re_branch.shape[0]
        N = xy_trunk.shape[1]

        branch_out = self.branch_net(re_branch)
        
        # Split output weights into psi and omega weights
        w_psi = branch_out[:, :self.p]
        w_omega = branch_out[:, self.p:]

        xy_flat = xy_trunk.reshape(-1, 2)
        trunk_out = self.trunk_net(xy_flat)
        trunk_out = trunk_out.reshape(B, N, 2 * self.p)
        
        t_psi = trunk_out[:, :, :self.p]
        t_omega = trunk_out[:, :, self.p:]

        # Inner products
        psi_pred = torch.einsum('bp,bnp->bn', w_psi, t_psi) + self.bias[0]
        omega_pred = torch.einsum('bp,bnp->bn', w_omega, t_omega) + self.bias[1]

        # Force psi = 0 at boundaries x=0, x=1, y=0, y=1
        x = xy_trunk[:, :, 0]
        y = xy_trunk[:, :, 1]
        boundary_factor = x * (1 - x) * y * (1 - y)
        
        # Streamfunction is 0 on walls
        psi_pred = psi_pred * boundary_factor

        # Stack outputs psi, omega
        return torch.stack([psi_pred, omega_pred], dim=-1)
    
def compute_residual(model, re, xy):
    """
    Laplacian(psi) + omega = 0
    (u.grad)omega - (1/Re)*Laplacian(omega) = 0
    """
    re.requires_grad_(True)
    xy.requires_grad_(True)

    # normalise Reynolds number over [0,1]
    re_norm = re / 1000.0
    
    # Forward pass
    preds = model(re_norm, xy)
    
    psi = preds[:, :, 0]
    omega = preds[:, :, 1]
    
    dpsi_dxy = torch.autograd.grad(psi, xy, 
                                  grad_outputs=torch.ones_like(psi), 
                                  create_graph=True, retain_graph=True)[0]
    dpsi_dx = dpsi_dxy[:, :, 0]
    dpsi_dy = dpsi_dxy[:, :, 1]
    
    domega_dxy = torch.autograd.grad(omega, xy, 
                                    grad_outputs=torch.ones_like(omega), 
                                    create_graph=True, retain_graph=True)[0]
    domega_dx = domega_dxy[:, :, 0]
    domega_dy = domega_dxy[:, :, 1]
    
    dpsi_xx = torch.autograd.grad(dpsi_dx, xy, 
                                 grad_outputs=torch.ones_like(dpsi_dx), 
                                 create_graph=True, retain_graph=True)[0][:, :, 0]
    dpsi_yy = torch.autograd.grad(dpsi_dy, xy, 
                                 grad_outputs=torch.ones_like(dpsi_dy), 
                                 create_graph=True, retain_graph=True)[0][:, :, 1]
    
    domega_xx = torch.autograd.grad(domega_dx, xy, 
                                   grad_outputs=torch.ones_like(domega_dx), 
                                   create_graph=True, retain_graph=True)[0][:, :, 0]
    domega_yy = torch.autograd.grad(domega_dy, xy, 
                                   grad_outputs=torch.ones_like(domega_dy), 
                                   create_graph=True, retain_graph=True)[0][:, :, 1]
    
    # Poisson equation: Laplac(psi) + omega = 0
    res_poisson = dpsi_xx + dpsi_yy + omega
    
    # u = dpsi/dy, v = -dpsi/dx
    u = dpsi_dy
    v = -dpsi_dx
    
    Re = re.expand(-1, xy.shape[1])
    
    transport_lhs = (u * domega_dx + v * domega_dy)
    transport_rhs = (1.0 / Re) * (domega_xx + domega_yy)
    
    res_vorticity = transport_lhs - transport_rhs

    # Corner masking for handling singularity points
    # Make top corners equal to zero
    x = xy[:, :, 0]
    y = xy[:, :, 1]

    # Mask radius 0.05 around (0,1) and (1,1)
    mask = torch.ones_like(res_poisson)
    corner_mask = (y > 0.95) & ((x < 0.05) | (x > 0.95))
    mask[corner_mask] = 0.0

    return res_poisson * mask, res_vorticity * mask