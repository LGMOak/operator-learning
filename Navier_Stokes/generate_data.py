import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

class CavityFlowSolver:
    def __init__(self, nx=64, ny=64, Lx=1.0, Ly=1.0):
        self.nx = nx
        self.ny = ny
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)

        # Grid
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def solve(self, Re, max_iter=10000, epsilon=1e-5):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        psi = np.zeros((ny, nx)) # Streaming function
        w = np.zeros((ny, nx)) # Vorticity

        denom = 2 * (1 / dx ** 2 + 1 / dy ** 2)

        for it in range(max_iter):
            psi_old = psi.copy()
            w_old = w.copy()

            # Solve streamfunction Poisson equation using central difference equation Jacobi iteration
            psi[1:-1, 1:-1] = ((psi[1:-1, 2:] + psi[1:-1, :-2]) / dx ** 2 +
                               (psi[2:, 1:-1] + psi[:-2, 1:-1]) / dy ** 2 +
                               w[1:-1, 1:-1]) / denom

            # Boundary condition for vorticity
            # Top wall moves
            w[-1, :] = -2 * psi[-2, :] / dy ** 2 - 2 / dy

            # rest are stationary
            w[0, :] = -2 * psi[1, :] / dy ** 2
            w[:, -1] = -2 * psi[:, -2] / dx ** 2
            w[:, 0] = -2 * psi[:, 1] / dx ** 2

            dpsi_dy = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dy)
            dpsi_dx = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dx)

            dw_dx = (w[1:-1, 2:] - w[1:-1, :-2]) / (2 * dx)
            dw_dy = (w[2:, 1:-1] - w[:-2, 1:-1]) / (2 * dy)

            # Laplacian
            lap_w = (w[1:-1, 2:] - 2 * w[1:-1, 1:-1] + w[1:-1, :-2]) / dx ** 2 + \
                    (w[2:, 1:-1] - 2 * w[1:-1, 1:-1] + w[:-2, 1:-1]) / dy ** 2

            # Update vorticity
            w[1:-1, 1:-1] = w[1:-1, 1:-1] + 0.001 * (
                    (1 / Re) * lap_w - (dpsi_dy * dw_dx - dpsi_dx * dw_dy)
            )

            # Check convergence
            diff = np.linalg.norm(w - w_old)
            if diff < epsilon:
                break
        return psi, w

    def generate_dataset(self, n_samples=1000):
        print(f"Generating {n_samples} Navier-Stokes samples...")

        # Inputs: Reynolds numbers
        Re_list = np.linspace(100, 1000, n_samples)

        dataset = {
            'Re': [],
            'psi': [],
            'w': [],
            'u': [],
            'v': [],
            'x': self.x,
            'y': self.y
        }

        pbar = tqdm(total=n_samples)

        for Re in Re_list:
            psi, w = self.solve(Re)

            # Calculate Velocity fields from Streamfunction
            # u = dpsi/dy, v = -dpsi/dx
            u = np.zeros_like(psi)
            v = np.zeros_like(psi)

            u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * self.dy)
            v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2 * self.dx)

            # Top lid correction
            u[-1, :] = 1.0

            dataset['Re'].append(Re)
            dataset['psi'].append(psi)
            dataset['w'].append(w)
            dataset['u'].append(u)
            dataset['v'].append(v)

            pbar.update(1)

        pbar.close()

        # Convert to arrays
        for k in ['Re', 'psi', 'w', 'u', 'v']:
            dataset[k] = np.array(dataset[k])

        # Save
        with open('./data/ns_data.pkl', 'wb') as f:
            pickle.dump(dataset, f)

        print("Dataset saved to ./data/ns_data.pkl")

        # Visualize one
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.contourf(self.X, self.Y, dataset['u'][-1], levels=50, cmap='RdBu_r')
        plt.title(f'U Velocity (Re={dataset["Re"][-1]:.1f})')
        plt.colorbar()

        plt.subplot(122)
        plt.contourf(self.X, self.Y, dataset['psi'][-1], levels=50, cmap='viridis')
        plt.title('Streamfunction')
        plt.colorbar()
        plt.savefig('./figures/ns_sample.png')
        print("Sample saved to ./figures/ns_sample.png")

if __name__ == "__main__":
    solver = CavityFlowSolver(nx=128, ny=128)
    solver.generate_dataset(n_samples=1000)