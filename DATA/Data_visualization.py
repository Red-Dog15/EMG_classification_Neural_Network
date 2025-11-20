import matplotlib.pyplot as plt
import torch

def plot_window(window):
    # window shape: (seq_len, channels)
    seq_len, channels = window.shape
    fig, axs = plt.subplots(channels, 1, figsize=(8, 2*channels), sharex=True)
    for c in range(channels):
        axs[c].plot(window[:, c])
        axs[c].set_ylabel(f"ch{c}")
    plt.xlabel("samples")
    plt.tight_layout()
    plt.show()
    
plot_window(10 * torch.rand(100, 8))  # example usage with random data