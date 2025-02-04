import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from torchvision import transforms
import networks
from PIL import Image

# Load models (specify paths to pre-trained MonoDepth2 models)
MODEL_PATHS = {
    "Monodepth2 Original": "models/mono_1024x320/",  # Example path
    "Depth Fourier Filtered": "models/mono_1024x320/fine_tuned/filtered_depth/combined/normalized/",
    "Depth-Dem Regularized": "models/mono_1024x320/fine_tuned/dem/combined/normalized/",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image
gt = True
index = 3
image_path = f"datasets/atlas-tiny/image/opt_000{index}.png"  # Specify your image path
#image_path = "/home/stesilve/Documents/github/pivot/dfvo/dataset/moonLanding/01_8bit/000001.png"
input_image = Image.open(image_path).convert('RGB')  # Open as RGB to ensure 3 channels (Monodepth2 expects this)

if gt:
    depth_path = f"datasets/atlas-tiny/dem/dem_000{index}.png"  # Specify your depth map path
    depth_gt = Image.open(depth_path).convert('I;16')  # Depth maps are single-channel
#depth_gt = np.array(depth_gt).astype(np.uint16) / 65535.0 * 128.0 # Normalize to [0, 1]

# Preprocess image
def preprocess_image(img, width=640, height=640):
    #img = cv2.resize(img, (width, height))
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,))
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
    input_tensor = preprocess_image(input_image)
    depth_maps[name] = infer_depth(name, encoder, depth_decoder, input_tensor)

# Convert depth to disparity
def depth_to_disparity(depth_map):
    return 1.0 / (depth_map + 1e-6)  # Avoid division by zero

# Plot results
if gt:
    fig, axes = plt.subplots(len(depth_maps), 4, figsize=(12, 8))
else:
    fig, axes = plt.subplots(len(depth_maps), 3, figsize=(12, 8))


for i, (name, depth) in enumerate(depth_maps.items()):
    disparity = depth_to_disparity(depth)
    
    if gt:
        axes[i, 0].imshow(input_image)
        axes[i, 0].set_title("Original Image")
        axes[i, 1].axis("off")
        axes[i, 2].axis("off")
        axes[i, 3].axis("off")
        
        axes[i, 1].imshow(depth_gt, cmap="magma",)# alpha=0.6)
        #axes[i, 3].imshow(depth, cmap="magma", alpha=0.4)
        axes[i, 1].set_title(f"Groundtruth")

        axes[i, 2].imshow(depth, cmap="plasma")
        axes[i, 2].set_title(f"{name} Depth Map")

        # axes[i, 1].imshow(disparity, cmap="magma")
        # axes[i, 1].set_title(f"{name} Disparity")

        axes[i, 3].imshow(input_image, alpha=0.6)
        axes[i, 3].imshow(depth, cmap="magma", alpha=0.4)
        axes[i, 3].set_title(f"Overlay")
    else:
        axes[i, 0].imshow(input_image)
        axes[i, 0].set_title("Original Image")
        axes[i, 1].axis("off")
        axes[i, 2].axis("off")
        
        axes[i, 1].imshow(depth, cmap="plasma")
        axes[i, 1].set_title(f"{name} Depth Map")

        # axes[i, 1].imshow(disparity, cmap="magma")
        # axes[i, 1].set_title(f"{name} Disparity")

        axes[i, 2].imshow(input_image, alpha=0.6)
        axes[i, 2].imshow(depth, cmap="magma", alpha=0.4)
        axes[i, 2].set_title(f"Overlay")

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
first_depth_map = list(depth_maps.values())[2]
input_image = np.asarray(input_image)

point_cloud = depth_to_point_cloud(first_depth_map, input_image)

o3d.visualization.draw_geometries([point_cloud])
