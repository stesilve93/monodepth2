import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import cv2


#DEBUG
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class DepthDataset(Dataset):
    def __init__(self, img_dir, depth_dir, img_size=(1024, 320), source="depth", normalize_maps=False):
        """
        Args:
            img_dir (string): Directory with all the images.
            depth_dir (string): Directory with all the depth maps.
            img_size (tuple): Desired image size (width, height) after resizing.
            source (string): Whether the depth maps are "depth" or "dem" values.

        """
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.img_size = img_size
        self.source = source
        self.normalize_maps = normalize_maps
        
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
            #transforms.Grayscale(num_output_channels=1),  # Ensure depth map is single channel
            #transforms.Resize(self.img_size),  # Resize depth map to 1024x320
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
        #extrema = image.getextrema()
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
        if self.source == "depth":
            depth = np.array(depth).astype(np.uint16) / 65535.0 * 128.0 # Normalize to [0, 1]
        else:
            depth = np.array(depth).astype(np.uint16) / 65535.0 # Normalize to [0, 1]            

        if self.normalize_maps:
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

        ####
        # plt.subplot(3, 2, 3), plt.imshow(depth, cmap='gray')
        # plt.title('Numpy image, original mode I'), plt.colorbar()
        # plt.subplot(3, 2, 4), plt.imshow(image, cmap='gray')
        # plt.title('Numpy image, original mode I'), plt.colorbar()
        ####

        # # Convert back to image
        image = Image.fromarray((image).astype(np.uint8))
        #depth = Image.fromarray((depth).astype(np.uint16))
        depth = depth.astype(np.float32) # The depth is treated as np float 32 to mantain the uint16 precision
        new_size = (640, 640)  # (width, height) in OpenCV
        # Resize using OpenCV
        depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_LINEAR)

        

        ### Data augmentation (commented for now)
        # Random augmentations (consistent for image and depth map)
        # if random.random() > 0.5:  # Horizontal flip
        #     image = transforms.functional.hflip(image)
        #     depth = transforms.functional.hflip(depth)
        # if random.random() > 0.5:  # Random small rotation
        #     angle = random.uniform(-5, 5)
        #     image = transforms.functional.rotate(image, angle)
        #     depth = transforms.functional.rotate(depth, angle)       


        # Apply transformations (resize and to tensor)
        image = self.transform_image(image)
        depth = self.transform_depth(depth)

        # depth_np = np.array(depth)
        # print(depth_np.dtype)  # Deve essere uint16
        # print(depth_np.min(), depth_np.max())  # Dovrebbe essere nel range 0-65535

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

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Sobel filter kernels
        sobel_x_kernel = torch.tensor([[-1, 0, 1], 
                                       [-2, 0, 2], 
                                       [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], 
                                       [ 0,  0,  0], 
                                       [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Load kernels into the convolution layers
        self.sobel_x.weight = nn.Parameter(sobel_x_kernel, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel, requires_grad=False)

    def forward(self, pred_depth, gt_depth):
        """
        pred_depth: Predicted depth map (B x 1 x H x W)
        gt_depth: Ground truth depth map (B x 1 x H x W)
        """
        # Move Sobel filters to the same device as input tensors
        self.sobel_x = self.sobel_x.to(pred_depth.device)
        self.sobel_y = self.sobel_y.to(pred_depth.device)

        # Compute gradients for predicted depth
        grad_pred_x = self.sobel_x(pred_depth)
        grad_pred_y = self.sobel_y(pred_depth)

        # Compute gradients for ground truth depth
        grad_gt_x = self.sobel_x(gt_depth)
        grad_gt_y = self.sobel_y(gt_depth)

        # Compute gradient differences (L1 loss on gradients)
        grad_diff_x = torch.abs(grad_pred_x - grad_gt_x)
        grad_diff_y = torch.abs(grad_pred_y - grad_gt_y)

        # Combine gradient losses
        edge_loss = torch.mean(grad_diff_x + grad_diff_y)

        return edge_loss

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, pred, target):
        # Apply a Gaussian filter for local statistics
        mu_pred = F.avg_pool2d(pred, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)
        mu_target = F.avg_pool2d(target, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)
        
        sigma_pred = F.avg_pool2d(pred ** 2, kernel_size=self.window_size, stride=1, padding=self.window_size // 2) - mu_pred ** 2
        sigma_target = F.avg_pool2d(target ** 2, kernel_size=self.window_size, stride=1, padding=self.window_size // 2) - mu_target ** 2
        sigma_pred_target = F.avg_pool2d(pred * target, kernel_size=self.window_size, stride=1, padding=self.window_size // 2) - mu_pred * mu_target

        # Compute SSIM
        ssim = ((2 * mu_pred * mu_target + self.c1) * (2 * sigma_pred_target + self.c2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + self.c1) * (sigma_pred + sigma_target + self.c2))
        
        # SSIM Loss (1 - SSIM)
        return 1 - ssim.mean()

class GradientMatchingLoss(nn.Module):
    def __init__(self):
        super(GradientMatchingLoss, self).__init__()
        # Sobel filters for x and y gradients
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        sobel_x_filter = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_filter = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_x.weight = nn.Parameter(sobel_x_filter, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_filter, requires_grad=False)

    def forward(self, pred, target):
        # Move Sobel filters to the same device as input tensors
        self.sobel_x = self.sobel_x.to(pred.device)
        self.sobel_y = self.sobel_y.to(pred.device)
        # Compute gradients
        grad_pred_x = self.sobel_x(pred)
        grad_pred_y = self.sobel_y(pred)
        grad_target_x = self.sobel_x(target)
        grad_target_y = self.sobel_y(target)

        # Compute gradient difference
        loss_x = torch.abs(grad_pred_x - grad_target_x).mean()
        loss_y = torch.abs(grad_pred_y - grad_target_y).mean()

        # Total Gradient Matching Loss
        return loss_x + loss_y

class CombinedLoss(nn.Module):
    def __init__(self, lambda_ssim=0.85, lambda_grad=0.15):
        super(CombinedLoss, self).__init__()
        self.ssim_loss = SSIMLoss()
        self.grad_loss = GradientMatchingLoss()
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad

    def forward(self, pred, target):
        ssim_loss = self.ssim_loss(pred, target)
        grad_loss = self.grad_loss(pred, target)
        return self.lambda_ssim * ssim_loss + self.lambda_grad * grad_loss

# DEBUG
# img_dir = "/home/mbussolino/Documents/Datasets/dataset_depth_00/imgs"  # Directory containing input images
# depth_dir = "/home/mbussolino/Documents/Datasets/dataset_depth_00/depth_maps" 
# save_path = "fine_tuned_model.pth"
# model_path = "models/mono_1024x320/"#../pivot/dfvo/model_zoo/depth/nyuv2/supervised/"
# log_dir = "runs/fine_tuning"  # Directory for TensorBoard logs

# # # Hyperparameters
# batch_size = 4
# learning_rate = 1e-5
# num_epochs = 10

# # # Dataset and Dataloader (using our custom DepthDataset)
# dataset = DepthDataset(img_dir, depth_dir, img_size=(640, 640), source='dem')
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# out = dataset.__getitem__(1)

