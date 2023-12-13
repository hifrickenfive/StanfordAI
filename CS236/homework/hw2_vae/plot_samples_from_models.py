import torch
from codebase.models.vae import VAE
from codebase.models.gmvae import GMVAE
from codebase.models.fsvae import FSVAE
from codebase.models.ssvae import SSVAE
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_model_by_name(model, global_step, device=None):
    """
    Load a model based on its name model.name and the checkpoint iteration step

    Args:
        model: Model: (): A model
        global_step: int: (): Checkpoint iteration
    """
    file_path = os.path.join(
        "checkpoints", model.name, "model-{:05d}.pt".format(global_step)
    )
    state = torch.load("C://data//StanfordAI//CS236//homework//vae//checkpoints//model=fsvae_run=0002//model-20000.pt", map_location=device)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model inputs
num_samples = 200
latent_dim = 10
num_checkpoint = 20000
model_choice = 'FSVAE'  # 'VAE', 'GMVAE', 'SSVAE_no_ELBO', 'SSAVE', 'FSVAE'

if model_choice == 'VAE':
    run_ID = '0002'
    model_name = "model=vae_z=10_run=" + run_ID
    model = VAE(z_dim=latent_dim, name=model_name).to(device)
elif model_choice == 'GMVAE':
    run_ID = '0000'
    model_name = "model=gmvae_z=10_k=500_run=" + run_ID
    model = GMVAE(z_dim=latent_dim, name=model_name).to(device)
elif model_choice == 'SSVAE_no_ELBO':
    run_ID = '0000' # No ELBO
    model_name = "model=ssvae_gw=000_cw=100_run=" + run_ID
    model = SSVAE(z_dim=latent_dim, name=model_name).to(device)
elif model_choice == 'SSVAE':
    run_ID = '0000' # ELBO
    model_name = "model=ssvae_gw=001_cw=100_run=" + run_ID
    model = SSVAE(z_dim=latent_dim, name=model_name).to(device)
elif model_choice == 'FSVAE':
    run_ID = '0001'
    latent_dim = 20
    num_checkpoint = 1000000
    model_name = "model=fsvae_run=" + run_ID
    model = FSVAE(name=model_name).to(device)

# Load model
load_model_by_name(model, num_checkpoint, device=device)

# Ensure the model is in evaluation mode
model.eval()
model = model.to(device)

if model_choice == 'FSVAE':
    # FSVAE sorted numerically and by latent variable
    num_samples = 200
    rows = 10
    columns = 20

    # Create labels y, where the ith row has all elements labeled as i
    y_indices = torch.cat([torch.full((columns,), i) for i in range(rows)])
    y = F.one_hot(y_indices, 10).float().to(device)

    # Create a grid for z that varies smoothly across columns
    z_start = torch.randn(rows, model.z_dim).to(device)
    z_end = torch.randn(rows, model.z_dim).to(device)
    alpha = torch.linspace(0, 1, columns).to(device)
    z_samples = [(1 - a) * z_start + a * z_end for a in alpha]
    z = torch.cat(z_samples, 0).to(device)

    with torch.no_grad():
        generated_samples = model.compute_mean_given(z, y)

    fig, axes = plt.subplots(10, 20, figsize=(20, 10))
    for ax, img in zip(axes.ravel(), generated_samples):
        img = img.view(3, 32, 32)  # Reshape the image to (3, 32, 32)
        ax.imshow(img.cpu().numpy().transpose(1, 2, 0))
        ax.axis('off')
else:
    with torch.no_grad():
        z = model.sample_z
        z = torch.randn((num_samples, latent_dim)).to(device)
        generated_samples = model.dec(z)
        generated_samples = torch.sigmoid(generated_samples)

    fig, axarr = plt.subplots(10, 20, figsize=(20, 10))
    for i in range(10):
        for j in range(20):
            ax = axarr[i, j]
            ax.imshow(generated_samples[i * 20 + j].cpu().reshape(28, 28), cmap='gray') # The other models
            ax.axis('off')

plt.tight_layout()
plt.savefig('generated_digits' + model_name + '.jpeg', format='jpeg', dpi=300)
