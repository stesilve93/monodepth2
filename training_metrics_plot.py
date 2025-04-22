import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Define folders
folders = ["test_mae", "test_rmse", "val_mae", "val_rmse"]
legend = ["DEM reg.: combined loss","DEM reg.: mse loss","DEM reg.: ssim loss",\
          "Fourier filt.: combined loss","Fourier filt.: edge loss","Fourier filt.: mse loss"]
titles = ["Test Set: Mean Absolute Error", "Test Set: Root-Mean Squared Error",\
          "Validation Set: Mean Absolute Error", "Validation Set: Root-Mean Squared Error"]
print(plt.style.available)
# Set up plot style
plt.style.use("seaborn-v0_8-paper")

matplotlib.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

custom_style_1 = (0, (7, 2,7,2,1,2))        
custom_style_2 = (0, (1, 2, 1, 2, 7, 2))  

line_styles = ['-', '--', '-.', ':',custom_style_1, custom_style_2]

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
            ax.plot(df["step"], df["value"], label=legend[j], linestyle=line_styles[j])
    
    ax.set_title(titles[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
# Collect legend handles and labels only from the first axs object
handles, labels = ax.get_legend_handles_labels()
# Adjust figure to create extra space above the subplots
fig.subplots_adjust(top=0.2)  # Increase top margin to accommodate the legend

# Place the global legend above the plot in 2 rows
fig.legend(handles, labels, loc='upper center', ncol=(len(labels) + 1) // 2,fontsize=14)


plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()