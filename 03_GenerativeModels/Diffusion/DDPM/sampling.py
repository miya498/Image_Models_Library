import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def sample_images(diffusion, model, num_samples, img_size, num_timesteps, device):
    model.eval()
    shape = (num_samples, 1, img_size, img_size)
    samples = diffusion.p_sample_loop(
        denoise_fn=model,
        shape=shape,
        noise_fn=torch.randn,
    )

    grid = make_grid(samples, nrow=5, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.show()