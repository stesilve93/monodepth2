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
import json
import pandas as pd


#DEBUG
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class CroppedDataset(Dataset):
    def __init__(self, img_dir, depth_dir, img_size=(320, 1024), source="depth", normalize_maps=False):
        """
        Args:
            img_dir (string): Directory with all the images.
            depth_dir (string): Directory with all the depth maps.
            img_size (tuple): Desired image size (width, height) after resizing.
            source (string): Whether the depth maps are "depth" or "dem" values.

        """
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.labels_dir = os.path.join(img_dir,'..','database_info','labels.json')
        self.img_size = img_size
        self.source = source
        self.normalize_maps = normalize_maps

        with open(self.labels_dir, 'r') as f:
            labels = json.load(f)
        self.labels = pd.DataFrame(labels)
        
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

        box = self.labels.iloc[idx]['bound-box']

        image = Image.open(img_path).convert('RGB')  # Open as RGB to ensure 3 channels (Monodepth2 expects this)
        depth = Image.open(depth_path).convert('I;16')  # Depth maps are single-channel


        # Convert to numpy arrays
        image = np.array(image)
        if self.source == "depth":
            depth = np.array(depth).astype(np.uint16) / 65535.0 * 128.0 # Normalize to [0, 1]
        else:
            depth = np.array(depth).astype(np.uint16) #/ 65535.0 # Normalize to [0, 1]            

        if self.normalize_maps:
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

        if box is not None:
            # Crop images on target
            expand = random.randint(10, 30)

            x_min, y_min, x_max, y_max = box
            x_min = max(0, x_min - expand)
            y_min = max(0, y_min - expand)
            x_max = min(image.shape[1], x_max + expand)
            y_max = min(image.shape[0], y_max + expand)

            # Crop image and depth map
            image = image[y_min:y_max, x_min:x_max]
            depth = depth[y_min:y_max, x_min:x_max]

        # # Convert back to image
        image = Image.fromarray((image).astype(np.uint8))
        depth = Image.fromarray((depth).astype(np.uint8))
        #depth = Image.fromarray(depth.astype(np.float32), mode='F')


        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # axs[0].imshow(image)
        # axs[0].set_title('RGB Image')
        # axs[0].axis('off')
        
        # im = axs[1].imshow(depth, cmap='plasma')
        # axs[1].set_title('Depth Map')
        # axs[1].axis('off')

        # fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
        # plt.tight_layout()
        # plt.show()

        ### Data augmentation
        # Random augmentations (consistent for image and depth map)
        if random.random() > 0.5:  # Horizontal flip
            image = transforms.functional.hflip(image)
            depth = transforms.functional.hflip(depth)
        # if random.random() > 0.5:  # Random small rotation
        #     angle = random.uniform(-5, 5)
        #     image = transforms.functional.rotate(image, angle)
        #     depth = transforms.functional.rotate(depth, angle)       


        # Apply transformations (resize and to tensor)
        image = self.transform_image(image)
        depth = self.transform_depth(depth)


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
    def __init__(self, lambda_ssim=0.85, lambda_grad=0.75):
        super(CombinedLoss, self).__init__()
        self.ssim_loss = SSIMLoss()
        self.grad_loss = GradientMatchingLoss()
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad

    def forward(self, pred, target):
        ssim_loss = self.ssim_loss(pred, target)
        grad_loss = self.grad_loss(pred, target)
        return self.lambda_ssim * ssim_loss + self.lambda_grad * grad_loss
    
class MaskLoss(nn.Module):
    def __init__(self, alpha = 2e3):
        super(MaskLoss, self).__init__()

        self.mse = nn.MSELoss() 
        self.combined = CombinedLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        
        comb_loss = self.combined(pred, target)

        # mask = target < 1  # Create boolean mask of valid pixels
        # pred_masked = pred[mask]
        # target_masked = target[mask]

        # if target_masked.numel() == 0:
        #     return comb_loss  # Avoid division by zero if no valid pixels

        # mse_loss = self.mse(pred_masked, target_masked)

        weights = 1.2 - target
        mse_loss = torch.mean(weights * (pred - target)**2)
        loss = self.alpha*mse_loss

        # print('Combined: ')
        # print(comb_loss)
        # print('Mse: ')
        # print(mse_loss)

        return loss

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

"""
Sampling strategies: RS (Random Sampling), EGS (Edge-Guided Sampling), and IGS (Instance-Guided Sampling)
"""
###########
# RANDOM SAMPLING
# input:
# inputs[i,:], targets[i, :], masks[i, :], self.mask_value, self.point_pairs
# return:
# inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B
###########
def randomSampling(inputs, targets, masks, threshold, sample_num):

    # find A-B point pairs from predictions
    inputs_index = torch.masked_select(inputs, targets.gt(threshold))
    num_effect_pixels = len(inputs_index)
    shuffle_effect_pixels = torch.randperm(num_effect_pixels).cuda()
    inputs_A = inputs_index[shuffle_effect_pixels[0:sample_num*2:2]]
    inputs_B = inputs_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # find corresponding pairs from GT
    target_index = torch.masked_select(targets, targets.gt(threshold))
    targets_A = target_index[shuffle_effect_pixels[0:sample_num*2:2]]
    targets_B = target_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # only compute the losses of point pairs with valid GT
    consistent_masks_index = torch.masked_select(masks, targets.gt(threshold))
    consistent_masks_A = consistent_masks_index[shuffle_effect_pixels[0:sample_num*2:2]]
    consistent_masks_B = consistent_masks_index[shuffle_effect_pixels[1:sample_num*2:2]]

    # The amount of A and B should be the same!!
    if len(targets_A) > len(targets_B):
        targets_A = targets_A[:-1]
        inputs_A = inputs_A[:-1]
        consistent_masks_A = consistent_masks_A[:-1]

    return inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B

###########
# EDGE-GUIDED SAMPLING
# input:
# inputs[i,:], targets[i, :], masks[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w
# return:
# inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B
###########
def ind2sub(idx, cols):
    r = idx / cols
    c = idx - r * cols
    return r, c

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx

def edgeGuidedSampling(inputs, targets, edges_img, thetas_img, masks, h, w):

    # find edges
    edges_max = edges_img.max()
    edges_mask = edges_img.ge(edges_max*0.1)
    edges_loc = edges_mask.nonzero()

    inputs_edge = torch.masked_select(inputs, edges_mask)
    targets_edge = torch.masked_select(targets, edges_mask)
    thetas_edge = torch.masked_select(thetas_img, edges_mask)
    minlen = inputs_edge.size()[0]

    # find anchor points (i.e, edge points)
    sample_num = minlen
    index_anchors = torch.randint(0, minlen, (sample_num,), dtype=torch.long).cuda()
    anchors = torch.gather(inputs_edge, 0, index_anchors)
    theta_anchors = torch.gather(thetas_edge, 0, index_anchors)
    row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)
    ## compute the coordinates of 4-points,  distances are from [2, 30]
    distance_matrix = torch.randint(2, 31, (4,sample_num)).cuda()
    pos_or_neg = torch.ones(4, sample_num).cuda()
    pos_or_neg[:2,:] = -pos_or_neg[:2,:]
    distance_matrix = distance_matrix.float() * pos_or_neg
    col = col_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.cos(theta_anchors).unsqueeze(0)).long()
    row = row_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.sin(theta_anchors).unsqueeze(0)).long()

    # constrain 0=<c<=w, 0<=r<=h
    # Note: index should minus 1
    col[col<0] = 0
    col[col>w-1] = w-1
    row[row<0] = 0
    row[row>h-1] = h-1

    # a-b, b-c, c-d
    a = sub2ind(row[0,:], col[0,:], w)
    b = sub2ind(row[1,:], col[1,:], w)
    c = sub2ind(row[2,:], col[2,:], w)
    d = sub2ind(row[3,:], col[3,:], w)
    A = torch.cat((a,b,c), 0)
    B = torch.cat((b,c,d), 0)

    inputs_A = torch.gather(inputs, 0, A.long())
    inputs_B = torch.gather(inputs, 0, B.long())
    targets_A = torch.gather(targets, 0, A.long())
    targets_B = torch.gather(targets, 0, B.long())
    masks_A = torch.gather(masks, 0, A.long())
    masks_B = torch.gather(masks, 0, B.long())

    return inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num

######################################################
# EdgeguidedRankingLoss (with regularization term)
# Please comment regularization_loss if you don't want to use multi-scale gradient matching term
#####################################################
class EdgeguidedRankingLoss(nn.Module):
    def __init__(self, point_pairs=10000, sigma=0.03, alpha=1.0, mask_value=-1e-8):
        super(EdgeguidedRankingLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.sigma = sigma # used for determining the ordinal relationship between a selected pair
        self.alpha = alpha # used for balancing the effect of = and (<,>)
        self.mask_value = mask_value
        #self.regularization_loss = GradientLoss(scales=4)

    def getEdge(self, images):
        n,c,h,w = images.size()
        a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        if c == 3:
            gradient_x = F.conv2d(images[:,0,:,:].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:,0,:,:].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x,2)+ torch.pow(gradient_y,2))
        edges = F.pad(edges, (1,1,1,1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1,1,1,1), "constant", 0)

        return edges, thetas

    def forward(self, inputs, targets, images, masks=None):
        if masks == None:
            masks = targets > self.mask_value
        # Comment this line if you don't want to use the multi-scale gradient matching term !!!
        # regularization_loss = self.regularization_loss(inputs.squeeze(1), targets.squeeze(1), masks.squeeze(1))
        # find edges from RGB
        edges_img, thetas_img = self.getEdge(images)

        #=============================
        n,c,h,w = targets.size()
        if n != 1:
            inputs = inputs.view(n, -1).double()
            targets = targets.view(n, -1).double()
            masks = masks.view(n, -1).double()
            edges_img = edges_img.view(n, -1).double()
            thetas_img = thetas_img.view(n, -1).double()

        else:
            inputs = inputs.contiguous().view(1, -1).double()
            targets = targets.contiguous().view(1, -1).double()
            masks = masks.contiguous().view(1, -1).double()
            edges_img = edges_img.contiguous().view(1, -1).double()
            thetas_img = thetas_img.contiguous().view(1, -1).double()

        # initialization
        loss = torch.DoubleTensor([0.0]).cuda()


        for i in range(n):
            # Edge-Guided sampling
            inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num = edgeGuidedSampling(inputs[i,:], targets[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w)
            # Random Sampling
            random_sample_num = sample_num
            random_inputs_A, random_inputs_B, random_targets_A, random_targets_B, random_masks_A, random_masks_B = randomSampling(inputs[i,:], targets[i, :], masks[i, :], self.mask_value, random_sample_num)

            # Combine EGS + RS
            inputs_A = torch.cat((inputs_A, random_inputs_A), 0)
            inputs_B = torch.cat((inputs_B, random_inputs_B), 0)
            targets_A = torch.cat((targets_A, random_targets_A), 0)
            targets_B = torch.cat((targets_B, random_targets_B), 0)
            masks_A = torch.cat((masks_A, random_masks_A), 0)
            masks_B = torch.cat((masks_B, random_masks_B), 0)
            
            # inputs_A = random_inputs_A
            # inputs_B = random_inputs_B
            # targets_A = random_targets_A
            # targets_B = random_targets_B
            # masks_A = random_masks_A
            # masks_B = random_masks_B

            #GT ordinal relationship
            target_ratio = torch.div(targets_A+1e-6, targets_B+1e-6)
            mask_eq = target_ratio.lt(1.0 + self.sigma) * target_ratio.gt(1.0/(1.0+self.sigma))
            labels = torch.zeros_like(target_ratio)
            labels[target_ratio.ge(1.0 + self.sigma)] = 1
            labels[target_ratio.le(1.0/(1.0+self.sigma))] = -1

            # consider forward-backward consistency checking, i.e, only compute losses of point pairs with valid GT
            consistency_mask = masks_A * masks_B

            equal_loss = (inputs_A - inputs_B).pow(2) * mask_eq.double() * consistency_mask
            unequal_loss = torch.log(1 + torch.exp((-inputs_A + inputs_B) * labels)) * (~mask_eq).double() * consistency_mask

            # Please comment the regularization term if you don't want to use the multi-scale gradient matching loss !!!
            loss = loss + self.alpha * equal_loss.mean() + 1.0 * unequal_loss.mean() + F.l1_loss(inputs,targets) #+ 0.2 * regularization_loss.double()

        return loss[0].float()/n
