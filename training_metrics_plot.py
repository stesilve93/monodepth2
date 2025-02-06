import os
import pandas as pd
import matplotlib.pyplot as plt

# Define folders
folders = ["test_mae", "test_rmse", "val_mae", "val_rmse"]
legend = ["DEM reg.: combined loss","DEM reg.: mse loss","DEM reg.: ssim loss",\
          "Fourier filt.: combined loss","Fourier filt.: edge loss","Fourier filt.: mse loss"]
titles = ["Test Set: Mean Absolute Error", "Test Set: Root-Mean Squared Error",\
          "Validation Set: Mean Absolute Error", "Validation Set: Root-Mean Squared Error"]
print(plt.style.available)
# Set up plot style
plt.style.use("seaborn-v0_8-paper")

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Iterate through each folder and plot the data
for i, folder in enumerate(folders):
    ax = axes[i]
    
    for j, file in enumerate(sorted(os.listdir(folder))):
        if file.endswith(".csv"):  # Ensure it's a CSV file
            filepath = os.path.join(folder, file)
            df = pd.read_csv(filepath, usecols=[1, 2], names=["step", "value"], header=0)
            ax.plot(df["step"], df["value"], label=legend[j])
    
    ax.set_title(titles[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()

plt.tight_layout()
plt.show()