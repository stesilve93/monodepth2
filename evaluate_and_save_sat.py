import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d as o3d
from torchvision import transforms
import networks
from PIL import Image
import random
import os

from datasets.cropped_dataset import CroppedDataset

# Load models (specify paths to pre-trained MonoDepth2 models)
MODEL_PATHS = {
    "MSE Train Original": "models/mono_1024x320/trained_env/sat/combined/cropped-augmented/", 
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLOT_FLAG = False

# Load image
gt = True
index = 7


img_dir = "/home/massi/Documents/datasets/trajectory_arc/images_bw"  # Directory containing input images
depth_dir = "/home/massi/Documents/datasets/trajectory_arc/depths/png_files"  # Directory containing ground truth depth maps
img_size = (320, 1024)  # Image dimensions

# Full dataset (this is the entire dataset, no split yet)
full_dataset = CroppedDataset(img_dir, depth_dir, img_size=img_size, source='sat', normalize_maps=True, crop_method = 'yolo')

for i in range(0,full_dataset.__len__()):
#for i in range(0,1):
    
    input_image = full_dataset.__getitem__(i)['image']  # Open as RGB to ensure 3 channels (Monodepth2 expects this)

    if gt:
        depth_gt = full_dataset.__getitem__(i)['depth']  # Depth maps are single-channel


    # Preprocess image
    def preprocess_image(img, width=640, height=640):
        #img = cv2.resize(img, (width, height))
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Resize((1024,320))
        ])
        return transform(img).unsqueeze(0).to(DEVICE)

    # Inference function
    def infer_depth(model, encoder, decoder, img_tensor):
        with torch.no_grad():
            features = encoder(img_tensor)
            outputs = decoder(features)
        depth = outputs[("disp", 0)].cpu().squeeze().numpy()
        return depth

    # Load models and infer depth
    depth_maps = {}
    for name, path in MODEL_PATHS.items():
        encoder = networks.ResnetEncoder(18, False).to(DEVICE)
        depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc).to(DEVICE)

        encoder.load_state_dict(torch.load(f"{path}/encoder.pth", map_location=DEVICE, weights_only=True), strict=False)
        depth_decoder.load_state_dict(torch.load(f"{path}/depth.pth", map_location=DEVICE, weights_only=True))

        encoder.eval()
        depth_decoder.eval()

        # Run inference
        
        #input_tensor = preprocess_image(input_image)
        input_tensor = input_image.unsqueeze(0).to(DEVICE)
        depth_maps[name] = infer_depth(name, encoder, depth_decoder, input_tensor)
        print(f'Inferred depth of image {i}')

    # Convert depth to disparity
    def depth_to_disparity(depth_map):
        return 1.0 / (depth_map + 1e-6)  # Avoid division by zero


    if PLOT_FLAG:
        # Plot results
        fig, axes = plt.subplots(len(depth_maps), 4 if gt else 3, figsize=(12, 8))
        axes = np.atleast_2d(axes)


        for i, (name, depth) in enumerate(depth_maps.items()):
            disparity = depth_to_disparity(depth)
            
            if gt:
                axes[i, 0].imshow(input_image.permute(1, 2, 0))
                axes[i, 0].set_title("Original Image")
                axes[i, 1].axis("off")
                axes[i, 2].axis("off")
                axes[i, 3].axis("off")
                
                axes[i, 1].imshow(depth_gt.permute(1, 2, 0), cmap="magma",)# alpha=0.6)
                # axes[i, 3].imshow(depth, cmap="magma", alpha=0.4)
                axes[i, 1].set_title(f"Groundtruth")

                axes[i, 2].imshow(depth, cmap="magma")
                axes[i, 2].set_title(f"{name} Depth Map")

                # axes[i, 1].imshow(disparity, cmap="magma")
                # axes[i, 1].set_title(f"{name} Disparity")

                axes[i, 3].imshow(input_image.permute(1, 2, 0), alpha=0.6)
                axes[i, 3].imshow(depth, cmap="magma", alpha=0.4)
                axes[i, 3].set_title(f"Overlay")
            else:
                axes[0].imshow(input_image)
                axes[0].set_title("Original Image")
                axes[1].axis("off")
                axes[2].axis("off")
                
                axes[1].imshow(depth, cmap="plasma")
                axes[1].set_title(f"{name} Depth Map")

                axes[i, 1].imshow(disparity, cmap="magma")
                axes[i, 1].set_title(f"{name} Disparity")

                axes[2].imshow(input_image, alpha=0.6)
                axes[2].imshow(depth, cmap="magma", alpha=0.4)
                axes[2].set_title(f"Overlay")

        plt.tight_layout()
        plt.show()

    # Convert depth map to point cloud
    def depth_to_point_cloud(depth, img):
        depth_map = np.asarray(depth)

        h, w = depth_map.shape
        fx, fy, cx, cy = w / 2, h / 2, w / 2, h / 2  # Approximate intrinsics
        print(depth_map.max())
        points = []
        colors = []
        for v in range(h):
            for u in range(w):
                z = depth_map[v, u]
                if z > 0:  # Filter invalid points
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append((x, y, z))
                    colors.append(img[v, u] / 255.0)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

        return point_cloud

    
    # Generate and visualize 3D point cloud for the first model
    first_depth_map = list(depth_maps.values())[0]
    input_image = np.asarray(input_image)

    # Salva la depth map
    output_depth_path = os.path.join("/home/massi/Documents/datasets/trajectory_arc/depths/inferred",f'frame-vis-{i:05d}.png') 
    depth_map_to_save = depth_maps[name]  # Depth map per il modello corrente

    # Normalizza la depth map e salvala come immagine
    depth_normalized = np.uint16(depth_map_to_save / depth_map_to_save.max() * 65535)  # Normalizza i valori tra 0 e 65535
    #depth_normalized = cv2.resize(depth_normalized, (1024,1024))
    cv2.imwrite(output_depth_path, full_dataset.restore_image(depth_normalized, i))

