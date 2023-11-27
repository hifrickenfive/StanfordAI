import argparse
import os

import tqdm
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from codebase.flow_network import MAF
from codebase.utils import make_halfmoon_toy_dataset, save_checkpoint, plot_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a MAF model.")
    parser.add_argument("--device", default="cpu", help="[cpu,cuda]")
    parser.add_argument(
        "--n_flows", default=5, type=int, help="number of planar flow layers"
    )
    parser.add_argument(
        "--hidden_size",
        default=100,
        type=int,
        help="number of hidden units in each flow layer",
    )
    parser.add_argument(
        "--n_epochs", default=50, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--n_samples",
        default=30000,
        type=int,
        help="total number of data points in toy dataset",
    )
    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    parser.add_argument("--out_dir", default="maf", help="path to output directory")
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(777)
    np.random.seed(777)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # load half-moons dataset
    train_loader, val_loader, test_loader = make_halfmoon_toy_dataset(
        args.n_samples, args.batch_size
    )

    # snippet of real data for plotting
    data_samples = test_loader.dataset

    # load model
    model = MAF(
        input_size=2, hidden_size=args.hidden_size, n_hidden=1, n_flows=args.n_flows
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    checkpoint = torch.load('maf/checkpoint.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])


    # save samples
    samples = model.sample(device, n=1000)
    plot_samples(samples, data_samples, 50, args)
