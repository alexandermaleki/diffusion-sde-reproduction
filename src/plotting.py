import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid


def denormalize(x):
    return (x / 2 + 0.5).clamp(0, 1)


def imshow(img, title=None):
    img = denormalize(img)
    npimg = img.detach().cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


def show_batch_from_loader(loader, title="Training images", nrow=8):
    images, _ = next(iter(loader))
    grid = torchvision.utils.make_grid(images[: nrow * nrow], nrow=nrow)
    imshow(grid, title=title)


def show_images(images, title=None, nrow=8):
    images = images.detach().cpu()
    grid = make_grid(images, nrow=nrow)
    imshow(grid, title=title)


def save_tensor_images_as_grid(images, filename, nrow=8):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    grid = make_grid(images.detach().cpu(), nrow=nrow)
    img = denormalize(grid).numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()

    print("Saved image grid:", filename)


def plot_loss(loss_hist, smooth_window=200, ylabel="VP score-matching loss"):
    plt.figure(figsize=(9, 4))

    if len(loss_hist) == 0:
        print("No loss history to plot.")
        return

    loss_arr = np.array(loss_hist)
    plt.plot(loss_arr, alpha=0.25, label="Raw loss")

    if smooth_window is not None and len(loss_arr) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(loss_arr, kernel, mode="valid")
        plt.plot(np.arange(smooth_window - 1, len(loss_arr)), smoothed, label=f"Smoothed loss, window={smooth_window}")

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title("Training loss")
    plt.grid(True)
    plt.legend()
    plt.show()
