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
    state = torch.load("C://data//StanfordAI//CS236//homework//vae//checkpoints//model=fsvae_run=0001//model-1000000.pt", map_location=device)
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

# with torch.no_grad():  # We don't need gradients for generation
    # Sample from the latent space
    # z = model.sample_z
    # z = torch.randn((num_samples, latent_dim)).to(device)
    # generated_samples = torch.sigmoid(generated_samples)

# fig, axarr = plt.subplots(10, 20, figsize=(20, 10))
# for i in range(10):
#     for j in range(20):
#         ax = axarr[i, j]
#         # ax.imshow(generated_samples[i * 20 + j].cpu().reshape(28, 28), cmap='gray') # The other models
#         ax.axis('off')

# FSVAE random label
# with torch.no_grad():
#     # 1. Sample latent variable z from its prior
#     z = model.sample_z(num_samples).to('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 2. Sample class labels y randomly (assuming labels are 0-indexed, i.e., 0 to 9)
#     y_indices = torch.randint(0, 10, (num_samples,)).to('cuda' if torch.cuda.is_available() else 'cpu')
#     y = F.one_hot(y_indices, 10).float().to('cuda' if torch.cuda.is_available() else 'cpu')  # One-hot encoding
    
#     # 3. Use the decoder to generate samples x given z and y
#     generated_samples = model.compute_mean_given(z, y)


# FSVAE sorted numerically
# num_samples = 200
# rows = 10
# columns = 20

# # Create labels y, where the ith row has all elements labeled as i
# y_indices = torch.cat([torch.full((columns,), i) for i in range(rows)])
# y = F.one_hot(y_indices, 10).float().to('cuda' if torch.cuda.is_available() else 'cpu')

# # Create z such that one of its dimensions varies according to column index j
# # For simplicity, let's say we are varying the first dimension of z by setting it to j (normalized to a reasonable range, e.g., [-2, 2])
# z = torch.randn(num_samples, model.z_dim).to('cuda' if torch.cuda.is_available() else 'cpu')  # start with random z
# for i in range(rows):
#     for j in range(columns):
#         z[i * columns + j, 0] = 4.0 * j / columns - 2.0  # This maps j=[0,19] to z[0]=-2 and z[19]=2

# # Now generate the samples
# with torch.no_grad():
#     generated_samples = model.compute_mean_given(z, y)

# FSVAE sorted numerically and by latent variable
num_samples = 200
rows = 10
columns = 20

# Create labels y, where the ith row has all elements labeled as i
y_indices = torch.cat([torch.full((columns,), i) for i in range(rows)])
y = F.one_hot(y_indices, 10).float().to('cuda' if torch.cuda.is_available() else 'cpu')

# Create a grid for z that varies smoothly across columns
z_start = torch.randn(rows, model.z_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
z_end = torch.randn(rows, model.z_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
alpha = torch.linspace(0, 1, columns).to('cuda' if torch.cuda.is_available() else 'cpu')
z_samples = [(1 - a) * z_start + a * z_end for a in alpha]
z = torch.cat(z_samples, 0).to('cuda' if torch.cuda.is_available() else 'cpu')

# Roll my own
linear_space = torch.linspace(-3, 3, 20)
z_jc = linear_space.repeat(200, 1).to('cuda' if torch.cuda.is_available() else 'cpu')

# Now generate the samples
with torch.no_grad():
    generated_samples = model.compute_mean_given(z, y)

fig, axes = plt.subplots(10, 20, figsize=(20, 10))
for ax, img in zip(axes.ravel(), generated_samples):
    img = img.view(3, 32, 32)  # Reshape the image to (3, 32, 32)
    ax.imshow(img.cpu().numpy().transpose(1, 2, 0))
    ax.axis('off')

plt.tight_layout()

# Save the figure to a JPEG file
plt.savefig('generated_digits' + model_name + '.jpeg', format='jpeg', dpi=300)
