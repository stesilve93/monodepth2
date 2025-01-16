import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import torch
import torch.nn as nn

#DEBUG
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class DepthDataset(Dataset):
    def __init__(self, img_dir, depth_dir, img_size=(1024, 320)):
        """
        Args:
            img_dir (string): Directory with all the images.
            depth_dir (string): Directory with all the depth maps.
            img_size (tuple): Desired image size (width, height) after resizing.
        """
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.img_size = img_size
        
        # List of all image and depth map filenames
        self.image_files = sorted(os.listdir(img_dir))
        self.depth_files = sorted(os.listdir(depth_dir))
        
        # Transformation for the images and depth maps (resize, to tensor, normalize)
        self.transform_image = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Ensure it's 3 channel (monodepth wants this)
            transforms.Resize(self.img_size),  # Resize image to 1024x320
            transforms.ToTensor(),  # Convert image to tensor
            #transforms.Normalize([0.5], [0.5])  # Normalize for input to Monodepth2 (scaled to [-1, 1])
        ])
        
        self.transform_depth = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure depth map is single channel
            transforms.Resize(self.img_size),  # Resize depth map to 1024x320
            transforms.ToTensor(),  # Convert to tensor
            #transforms.Normalize([0.5], [0.5])  # Normalize for input to Monodepth2 (scaled to [-1, 1])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load the image and depth map
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])

        image = Image.open(img_path).convert('RGB')  # Open as RGB to ensure 3 channels (Monodepth2 expects this)
        depth = Image.open(depth_path).convert('I;16')  # Depth maps are single-channel

        ####
        # min_val, max_val = depth.getextrema()
        # extrema = image.getextrema()
        # print(min_val, max_val)
        # extrema = image.getextrema()
        # print(extrema[0][0], extrema[0][1])
        # plt.figure(0, figsize=(18, 5))
        # min_val, max_val = depth.getextrema()
        # plt.subplot(3, 2, 1), plt.imshow(depth, vmin=min_val, vmax=max_val, cmap='gray')
        # plt.title('Pillow image, original mode I'), plt.colorbar()
        # plt.subplot(3, 2, 2), plt.imshow(image, vmin=extrema[0][0], vmax=extrema[0][1], cmap='gray')
        # plt.title('Pillow image, original mode I'), plt.colorbar()
        #####

        # Convert to numpy arrays
        image = np.array(image)
        depth = np.array(depth).astype(np.uint16) / 65535.0 * 128.0 # Normalize to [0, 1]

        ####
        # plt.subplot(3, 2, 3), plt.imshow(depth, cmap='gray')
        # plt.title('Numpy image, original mode I'), plt.colorbar()
        # plt.subplot(3, 2, 4), plt.imshow(image, cmap='gray')
        # plt.title('Numpy image, original mode I'), plt.colorbar()
        ####

        # # Convert back to image
        image = Image.fromarray((image).astype(np.uint8))
        depth = Image.fromarray((depth).astype(np.uint8))

        # Apply transformations (resize and to tensor)
        image = self.transform_image(image)
        depth = self.transform_depth(depth)

        # plt.subplot(3, 2, 5), plt.imshow(depth_t.permute(1, 2, 0),vmin=torch.min(depth_t), vmax=torch.max(depth_t), cmap='gray')
        # plt.title('Tensor show'), plt.colorbar()
        # plt.subplot(3, 2, 6), plt.imshow(image_t.permute(1, 2, 0),vmin=torch.min(image_t), vmax=torch.max(image_t), cmap='gray')
        # plt.title('Tensor show'), plt.colorbar()

        # plt.show()
        # Return a dictionary with image and depth tensor
        return {'image': image, 'depth': depth}

class ScaleInvariantLoss(nn.Module):
    def __init__(self):
        super(ScaleInvariantLoss, self).__init__()
    
    def forward(self, predicted, ground_truth):
        # Apply log to predicted and ground truth depth maps
        log_pred = torch.log(predicted + 1e-8)  # Adding epsilon to avoid log(0)
        log_gt = torch.log(ground_truth + 1e-8)
        
        # Compute d_i (difference between predicted and ground truth log depths)
        d = log_pred - log_gt
        
        # Compute the two terms in the loss
        n = d.numel()  # Total number of pixels
        term1 = torch.sum(d ** 2) / n
        term2 = (torch.sum(d) ** 2) / (n ** 2)
        
        # Scale-invariant loss
        loss = term1 - term2
        
        return loss


# DEBUG
# img_dir = "../datasets/atlas-tiny/image/"
# depth_dir = "../datasets/atlas-tiny/depth/"
# save_path = "fine_tuned_model.pth"
# model_path = "models/mono_1024x320/"#../pivot/dfvo/model_zoo/depth/nyuv2/supervised/"
# log_dir = "runs/fine_tuning"  # Directory for TensorBoard logs

# # # Hyperparameters
# batch_size = 4
# learning_rate = 1e-5
# num_epochs = 10

# # # Dataset and Dataloader (using our custom DepthDataset)
# dataset = DepthDataset(img_dir, depth_dir, img_size=(640, 640))
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# out = dataset.__getitem__(1)

