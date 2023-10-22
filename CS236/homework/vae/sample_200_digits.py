import torch
from codebase.models.vae import VAE
from codebase.models.gmvae import GMVAE
from codebase.models.ssvae import SSVAE
import matplotlib.pyplot as plt
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
    state = torch.load(file_path, map_location=device)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model inputs
num_samples = 200
latent_dim = 10
num_checkpoint = 20000
model_choice = 'GMVAE'  # choices are 'VAE', 'GMVAE', 'SSVAE'
run_ID = '0000'

if model_choice == 'VAE':
    model_name = "model=vae_z=10_run=" + run_ID
    model = VAE(z_dim=latent_dim, name=model_name).to(device)
elif model_choice == 'GMVAE':
    model_name = "model=gmvae_z=10_k=500_run=" + run_ID
    model = GMVAE(z_dim=latent_dim, name=model_name).to(device)

# Load model
load_model_by_name(model, num_checkpoint, device=device)

# Ensure the model is in evaluation mode
model.eval()
model = model.to(device)

with torch.no_grad():  # We don't need gradients for generation
    # Sample from the latent space
    z = torch.randn((num_samples, latent_dim)).to(device) 
    # Decode the samples
    generated_samples = model.dec(z)
    generated_samples = torch.sigmoid(generated_samples)

fig, axarr = plt.subplots(10, 20, figsize=(20, 10))

for i in range(10):
    for j in range(20):
        ax = axarr[i, j]
        ax.imshow(generated_samples[i * 20 + j].cpu().reshape(28, 28), cmap='gray')
        ax.axis('off')

plt.tight_layout()

# Save the figure to a JPEG file
plt.savefig('generated_digits' + model_name + '.jpeg', format='jpeg', dpi=300)
