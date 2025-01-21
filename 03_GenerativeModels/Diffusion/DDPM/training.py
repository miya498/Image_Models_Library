import torch
from torch.optim import Adam
from tqdm import tqdm

def train_model(model, diffusion, dataloader, epochs, num_timesteps, lr, device):
    optimizer = Adam(model.parameters(), lr=lr)
    model.to(device)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            t = torch.randint(1, num_timesteps + 1, (images.size(0),), device=device)
            noise = torch.randn_like(images).to(device)

            x_noisy = diffusion.q_sample(x_start=images, t=t, noise=noise)
            noise_pred = model(x_noisy, t)

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataloader))
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {losses[-1]:.4f}")

    return losses