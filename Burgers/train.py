import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import matplotlib.pyplot as plt
from deeponet import DeepONet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 2000
LEARNING_RATE = 1e-3
DATA_PATH = './data/burgers_data_train.pkl'

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    u0 = data['initial_conditions']
    x = data['x']
    t = data['t']

    T_grid, X_grid = np.meshgrid(t, x, indexing='ij')

    xt_combined = np.stack([X_grid.flatten(), T_grid.flatten()], axis=-1)

    xt_train = np.tile(xt_combined, (u0.shape[0], 1, 1))

    y_train = data['solutions'].reshape(u0.shape[0], -1)

    return (torch.tensor(u0, dtype=torch.float32), torch.tensor(xt_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32))

def train():
    print(f"Training on {device}")

    print("Loading data...")
    u0, xt, y = load_data(DATA_PATH)

    dataset = TensorDataset(u0, xt, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # branch_input_size must match data spatial resolution
    nx_res = u0.shape[1]

    model = DeepONet(branch_input_size=nx_res, hidden_dim=128, p=128).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=100, verbose=True)

    # Training loop
    loss_history = []

    print("Training started")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_u0, batch_xt, batch_y in loader:
            batch_u0 = batch_u0.to(device)
            batch_xt = batch_xt.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_u0, batch_xt)

            loss = loss_fn(pred, batch_y)

            # backwards pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        # average loss
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        scheduler.step(avg_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.6f}")

    # Save model
    torch.save(model.state_dict(), './models/deeponet_burgers.pth')
    print("Model saved successfully.")

    # Plot Loss
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title('Training Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./figures/training_loss.png')

    return model

if __name__ == "__main__":
    train()