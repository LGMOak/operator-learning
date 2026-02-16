import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PI_DeepONet.pi_deeponet import PIDeepONet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print("Loading test data")
    with open('../data/burgers_data_test.pkl', 'rb') as f:
        data = pickle.load(f)

    u0_test = data['initial_conditions']
    u_sol_test = data['solutions']
    x = data['x']
    t = data['t']

    nx_res = u0_test.shape[1]

    # model = DeepONet(branch_input_size=nx_res, hidden_dim=128, p=128)
    model = PIDeepONet(branch_input_size=nx_res, hidden_dim=128, p=128)
    model.load_state_dict(torch.load('./models/pi_deeponet_burgers.pth', map_location=device))
    model.to(device)
    model.eval()

    # Predict on a few random samples
    indices = np.random.choice(len(u0_test), 3, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for i, idx in enumerate(indices):
        u0_sample = torch.tensor(u0_test[idx:idx + 1], dtype=torch.float32).to(device)

        # Ground Truth
        u_true = u_sol_test[idx]

        # Prediction
        u_pred_tensor = model.predict(u0_sample, x, t)
        u_pred = u_pred_tensor.cpu().numpy().squeeze()

        # Error
        error = np.abs(u_true - u_pred)

        # Exact
        im1 = axes[i, 0].imshow(u_true.T, aspect='auto', origin='lower',
                                extent=[0, 1, -1, 1], cmap='RdBu_r', vmin=-2, vmax=2)
        axes[i, 0].set_title(f"Ground Truth (Sample {idx})")
        plt.colorbar(im1, ax=axes[i, 0])

        # Prediction
        im2 = axes[i, 1].imshow(u_pred.T, aspect='auto', origin='lower',
                                extent=[0, 1, -1, 1], cmap='RdBu_r', vmin=-2, vmax=2)
        axes[i, 1].set_title(f"DeepONet Prediction")
        plt.colorbar(im2, ax=axes[i, 1])

        # Error
        im3 = axes[i, 2].imshow(error.T, aspect='auto', origin='lower',
                                extent=[0, 1, -1, 1], cmap='viridis')
        axes[i, 2].set_title(f"Absolute Error (Max: {error.max():.4f})")
        plt.colorbar(im3, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig('./figures/pi_deeponet_evaluation.png')
    print("Evaluation saved to pi_deeponet_evaluation.png")

if __name__ == "__main__":
    evaluate()