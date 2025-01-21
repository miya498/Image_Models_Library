import os
import pickle
import time
import json
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from scipy.stats import entropy

from .classifier_metrics_numpy import classifier_score_from_logits, frechet_classifier_distance_from_activations

# ===== Utilities =====
def normalize_data(x):
    """Normalize input data to [-1, 1]."""
    return x / 127.5 - 1.0

def unnormalize_data(x):
    """Unnormalize data from [-1, 1] to [0, 255]."""
    return (x + 1.0) * 127.5

def seed_all(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ===== Inception Model Utilities =====
class InceptionModel:
    """Helper for running InceptionV3 for feature extraction."""

    def __init__(self, device="cuda"):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def run(self, images):
        """
        Run images through the Inception model and return the pool3 features and logits.
        """
        images = TF.resize(images, (299, 299))
        logits = self.model(images)
        pool3 = logits[:, :-1]  # Drop the classification layer
        return {"pool_3": pool3.cpu().numpy(), "logits": logits.cpu().numpy()}

class Model:
    """
    Base class for training and sampling models.
    All images (inputs and outputs) should be normalized to [-1, 1].
    """

    def train_fn(self, x, y):
        """Define training logic. Returns a dictionary with 'loss'."""
        raise NotImplementedError

    def samples_fn(self, dummy_x, y):
        """Generate samples. Returns a dictionary of outputs."""
        raise NotImplementedError

    def sample_and_run_inception(self, dummy_x, y, clip_samples=True):
        """
        Generate samples and compute Inception features.
        """
        samples_dict = self.samples_fn(dummy_x, y)
        results = {}
        for k, v in samples_dict.items():
            if clip_samples:
                v = torch.clamp(v, -1.0, 1.0)
            results[k] = self.inception_model.run(unnormalize_data(v))
        return results

    def bpd_fn(self, x, y):
        """Compute Bits-Per-Dimension (optional)."""
        return None

class EMA:
    """
    Helper class for Exponential Moving Average of model parameters.
    """
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self):
        """Update the shadow parameters with the current model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.shadow_params[name].data = self.decay * self.shadow_params[name].data + (1 - self.decay) * param.data

    def apply(self):
        """Apply the shadow parameters to the model."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(self.shadow_params[name].data)

    def restore(self):
        """Restore the original model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(self.shadow_params[name].data)

class Trainer:
    """
    Training logic for PyTorch models.
    """

    def __init__(self, model, optimizer, lr_scheduler, ema=None, log_dir="./logs"):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ema = ema
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_step(self, batch):
        """
        Perform a single training step.
        """
        self.model.train()
        images, labels = batch
        images = normalize_data(images)

        # Forward pass and loss computation
        outputs = self.model.train_fn(images, labels)
        loss = outputs["loss"]

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update EMA
        if self.ema is not None:
            self.ema.update()

        return loss.item()

    def evaluate(self, data_loader, prefix="eval"):
        """
        Evaluate the model and log results to TensorBoard.
        """
        self.model.eval()
        metrics = {}
        for batch in data_loader:
            images, labels = batch
            images = normalize_data(images)

            with torch.no_grad():
                outputs = self.model.samples_fn(images, labels)
                # Compute Inception features and metrics
                inception_features = self.model.sample_and_run_inception(images, labels)
                metrics.update(self.compute_metrics(inception_features))

        # Log metrics
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}/{k}", v)
        self.writer.flush()

    def compute_metrics(self, inception_features):
        """
        Compute FID and Inception scores from features.
        """
        metrics = {}
        logits = inception_features["logits"]
        pool_3 = inception_features["pool_3"]

        # Compute Inception score
        metrics["inception_score"] = classifier_score_from_logits(logits)

        # Compute FID (requires real data activations)
        # Assuming `real_activations` is available
        if hasattr(self, "real_activations"):
            metrics["fid"] = frechet_classifier_distance_from_activations(
                self.real_activations, pool_3
            )
        return metrics

class Sampler:
    """
    Generate samples and compute metrics for evaluation.
    """

    def __init__(self, model, dataset, log_dir="./samples", num_samples=50000, batch_size=64, device="cuda"):
        self.model = model
        self.dataset = dataset
        self.log_dir = log_dir
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device

        os.makedirs(log_dir, exist_ok=True)

    def generate_samples(self, ema=False):
        """
        Generate samples using the model.
        """
        self.model.eval()
        if ema:
            self.model.ema.apply()

        samples = []
        with torch.no_grad():
            for _ in trange(self.num_samples // self.batch_size, desc="Generating Samples"):
                dummy_x = torch.zeros((self.batch_size, *self.dataset.image_shape), device=self.device)
                y = torch.randint(0, self.dataset.num_classes, (self.batch_size,), device=self.device)
                batch_samples = self.model.samples_fn(dummy_x, y)
                samples.append(batch_samples)

        if ema:
            self.model.ema.restore()

        # Concatenate results
        concatenated = {
            key: torch.cat([batch[key] for batch in samples], dim=0)[:self.num_samples].cpu().numpy()
            for key in samples[0].keys()
        }
        return concatenated

    def save_samples(self, samples, step, prefix="sample"):
        """
        Save generated samples to disk.
        """
        filename = os.path.join(self.log_dir, f"{prefix}_step{step:09d}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Samples saved to {filename}")

    def evaluate_samples(self, samples, real_features):
        """
        Compute FID and Inception Score for the generated samples.
        """
        logits = samples["logits"]
        pool_3 = samples["pool_3"]

        # Compute Inception Score
        inception_score = classifier_score_from_logits(logits)

        # Compute FID
        fid = frechet_classifier_distance_from_activations(real_features, pool_3)

        return {"inception_score": inception_score, "fid": fid}

class CheckpointManager:
    """
    Helper class for saving and loading checkpoints.
    """

    def __init__(self, model, optimizer, checkpoint_dir="./checkpoints"):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, step):
        """
        Save the model and optimizer state.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_step{step:09d}.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load the model and optimizer state from a checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]
        print(f"Checkpoint loaded from {checkpoint_path}, step {step}")
        return step

class DistributedSampler:
    """
    Simplified distributed data processing for multi-GPU setups.
    """

    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device

    def distributed_fn(self, fn, *args, reduction="mean"):
        """
        Apply a function in a distributed setting and reduce results.
        """
        results = fn(*args)
        if reduction == "mean":
            return torch.mean(results, dim=0)
        elif reduction == "concat":
            return torch.cat(results, dim=0)
        else:
            raise ValueError(f"Unsupported reduction method: {reduction}")

def main():
    # Model and optimizer setup
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    ema = EMA(model, decay=0.9999)

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(model, optimizer)

    # Dataset and dataloaders
    train_loader = ...  # Setup PyTorch DataLoader for training
    eval_loader = ...   # Setup PyTorch DataLoader for evaluation

    # Training
    trainer = Trainer(model, optimizer, lr_scheduler, ema)
    for step in range(10000):
        for batch in train_loader:
            loss = trainer.train_step(batch)
            print(f"Step {step}, Loss: {loss:.4f}")

        if step % 1000 == 0:
            trainer.evaluate(eval_loader, prefix=f"step_{step}")
            checkpoint_manager.save_checkpoint(step)

    # Sampling
    sampler = Sampler(model, dataset=...)
    samples = sampler.generate_samples(ema=True)
    sampler.save_samples(samples, step=10000)