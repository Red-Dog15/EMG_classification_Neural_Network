import matplotlib.pyplot as plt
import torch
import os
import Data_Conversion as DC

def plot_window(window, file_name):
    # window shape: (seq_len, channels)
    seq_len, channels = window.shape
    fig, axs = plt.subplots(channels, 1, figsize=(8, 2*channels), sharex=True)
    for c in range(channels):
        axs[c].plot(window[:, c])
        axs[c].set_ylabel(f"ch{c}")
    plt.xlabel("samples")
    plt.tight_layout()
        # Save to results/
    plt.savefig(f"DATA/results/{file_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.show()
    
# Ensure folder exists
os.makedirs("DATA/results", exist_ok=True)
plot_window(10 * torch.rand(100, 8), "random_generated_data")  # example usage with random data

plot_window(DC.tensor_Light_Hand_Open, "Low_intensity_data")  # example usage tensor Light movements
plot_window(DC.tensor_Medium_Hand_Open, "Medium_intensity_data")  # example usage tensor Medium movements
plot_window(DC.tensor_Hard_Hand_Open, "Hard_intensity_data") # example usage tensor Hard movements
"""
# 
plot_window(DC.tensors_dict["Light"][0][7], "Low_intensity_data")  # example usage tensor Light movements
plot_window(DC.tensors_dict["Medium"][0][7], "Medium_intensity_data")  # example usage tensor Medium movements
plot_window(DC.tensors_dict["Hard"][0][7], "Hard_intensity_data") # example usage tensor Hard movements
"""