import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

class DataGenerator:
    def __init__(self, nu=0.01/np.pi, nx=256, nt=201, L=2.0, T=1.0):
        """
        :param nu: Viscosity parameter
        :param nx: spatial resolution
        :param nt: temporal resolution
        :param L: spatial domain length
        :param T: time horizon
        """
        self.nu = nu
        self.nx = nx
        self.nt = nt
        self.L = L
        self.T = T

        # spatial grid
        self.x = np.linspace(-L/2, L/2, nx, endpoint=False)
        self.dx = L / nx

        # temporal grid
        self.t = np.linspace(0, T, self.nt)

        # wavenumber
        # self.k = 2 * np.pi * np.fft.fftfreq(self.nt, d=self.dx)

    def generate_random_ic(self, ic_type='fourier'):
        """
        Generate random initial conditions
        Types:
        - fourier: random Fourier series
        - gaussian: sum of Gaussians
        - shock: discontinuous functions
        Want to learn on any wave shape u(t=0) to predict u(t=1)
        """
        x = self.x

        if ic_type == 'fourier':
            # random sine and cosine initial waves
            n_modes = np.random.randint(1, 6)
            u0 = np.zeros_like(x)

            for _ in range(n_modes):
                freq = np.random.randint(1, 8)
                amp = np.random.randn() * 0.5
                phase = np.random.uniform(0, 2*np.pi)

                if np.random.rand() > 0.5:
                    u0 += amp * np.sin(freq * np.pi * x / self.L + phase)
                else:
                    u0 += amp * np.cos(freq * np.pi * x / self.L + phase)

        elif ic_type == 'gaussian':
            # sum of Gaussian
            n_peaks = np.random.randint(1, 5)
            u0 = np.zeros_like(x)

            for _ in range(n_peaks):
                centre = np.random.uniform(-self.L/2, self.L/2)
                width = np.random.uniform(0.1, 0.5)
                amp = np.random.randn() * 2.0
                u0 += amp * np.exp(-((x - centre)/width)**2)

        elif ic_type == 'shock':
            # piecewise constant functions
            n_pieces = np.random.randint(2, 6)
            edges = np.sort(np.random.uniform(-self.L/2, self.L/2, n_pieces))
            edges = np.concatenate([[-self.L/2], edges, [self.L/2]])

            u0 = np.zeros_like(x)
            for i in range(len(edges) - 1):
                mask = (x >= edges[i]) & (x < edges[i+1])
                u0[mask] = np.random.uniform(-2, 2)

        return u0

    def solve_burgers(self, u0):
        """
        Solves u_t + u*u_x = nu*u_xx
        """

        def burger_rhs(t, u):
            # Finite Difference for derivatives (assuming Periodic BCs)
            u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * self.dx)
            u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (self.dx ** 2)

            dt = self.nu * u_xx - u * u_x
            return dt

        solution = solve_ivp(
            fun=burger_rhs,
            t_span=(0, self.T),
            y0=u0,
            t_eval=self.t,  # Output at the exact times we need
            method='RK45'  # Standard Runge-Kutta solver
        )

        # Shape: [Time, Space]
        return solution.y.T

    def generate_dataset(self, n_functions=1000, ic_types=['fourier', 'gaussian', 'shock'],
                         save_path='./data/burgers_data_train.pkl'):
        """
        Generate full dataset of IC-solution pairs
        """
        print(f"Generating {n_functions} Burgers solutions...")

        dataset = {
            'initial_conditions': [],
            'solutions': [],
            'x': self.x,
            't': self.t,
            'nu': self.nu,
            'metadata': []
        }

        #  get exactly n_functions valid samples
        count = 0
        pbar = tqdm(total=n_functions)

        while count < n_functions:
            # Randomly select IC type
            ic_type = np.random.choice(ic_types)

            # Generate IC and solve
            u0 = self.generate_random_ic(ic_type=ic_type)
            u_sol = self.solve_burgers(u0)

            # expect shape (nt, nx). If it's shorter, the solver failed.
            if u_sol.shape[0] == self.nt:
                dataset['initial_conditions'].append(u0)
                dataset['solutions'].append(u_sol)
                dataset['metadata'].append({'ic_type': ic_type})

                count += 1
                pbar.update(1)
            else:
                pass

        pbar.close()

        # Convert to arrays
        dataset['initial_conditions'] = np.array(dataset['initial_conditions'])
        dataset['solutions'] = np.array(dataset['solutions'])

        # Save
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"Dataset saved to {save_path}")
        print(f"Shape: {dataset['initial_conditions'].shape} -> {dataset['solutions'].shape}")

        return dataset

    def visualise_samples(self, dataset, n_samples=4):
        """
        Visualise some random samples from dataset
        """
        indices = np.random.choice(len(dataset['initial_conditions']), n_samples, replace=False)

        fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3 * n_samples))

        for idx, i in enumerate(indices):
            u0 = dataset['initial_conditions'][i]
            u_sol = dataset['solutions'][i]

            # Plot IC
            axes[idx, 0].plot(self.x, u0, 'b-', linewidth=2)
            axes[idx, 0].set_xlabel('x')
            axes[idx, 0].set_ylabel('u(x, 0)')
            axes[idx, 0].set_title(f"IC #{i} ({dataset['metadata'][i]['ic_type']})")
            axes[idx, 0].grid(True, alpha=0.3)

            # Plot solution heatmap
            im = axes[idx, 1].imshow(u_sol.T, aspect='auto', origin='lower',
                                     extent=[0, self.T, -self.L / 2, self.L / 2],
                                     cmap='RdBu_r', vmin=-2, vmax=2)
            axes[idx, 1].set_xlabel('t')
            axes[idx, 1].set_ylabel('x')
            axes[idx, 1].set_title(f"Solution u(x,t)")
            plt.colorbar(im, ax=axes[idx, 1])

        plt.tight_layout()
        plt.savefig('./figures/dataset_samples.png', dpi=150, bbox_inches='tight')
        print("Saved dataset_samples.png")


if __name__ == "__main__":
    generator = DataGenerator(
        nu=0.01 / np.pi,
        nx=256,
        nt=201,
        L=2.0,
        T=1.0
    )

    # Generate Training Data
    dataset = generator.generate_dataset(
        n_functions=1000,
        ic_types=['fourier', 'gaussian', 'shock'],
        save_path='./data/burgers_data_train.pkl'
    )

    # Generate Test Data
    print("\nGenerating test data...")
    test_dataset = generator.generate_dataset(
        n_functions=100,
        ic_types=['fourier', 'gaussian', 'shock'],
        save_path='./data/burgers_data_test.pkl'
    )

    # 4. Visualise
    generator.visualise_samples(dataset, n_samples=6)