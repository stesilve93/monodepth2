import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from torchvision import transforms
import networks

# Load models (specify paths to pre-trained MonoDepth2 models)
MODEL_PATHS = {
    "Depth": "models/mono_1024x320/fine_tuned/depth/",  # Example path
    "Depth-Dem Regularized": "models/mono_1024x320/fine_tuned/dem/",
    "Depth Fourier Filtered": "models/mono_1024x320/fine_tuned/filtered_depth/",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image
image_path = "datasets/"  # Specify your image path
input_image = cv2.imread(image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Preprocess image
def preprocess_image(img, width=640, height=640):
    img = cv2.resize(img, (width, height))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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

    encoder.load_state_dict(torch.load(f"{path}/encoder.pth", map_location=DEVICE))
    depth_decoder.load_state_dict(torch.load(f"{path}/depth.pth", map_location=DEVICE))

    encoder.eval()
    depth_decoder.eval()

    # Run inference
    input_tensor = preprocess_image(input_image)
    depth_maps[name] = infer_depth(name, encoder, depth_decoder, input_tensor)

# Convert depth to disparity
def depth_to_disparity(depth_map):
    return 1.0 / (depth_map + 1e-6)  # Avoid division by zero

# Plot results
fig, axes = plt.subplots(len(depth_maps) + 1, 3, figsize=(12, 8))

axes[0, 0].imshow(input_image)
axes[0, 0].set_title("Original Image")
axes[0, 1].axis("off")
axes[0, 2].axis("off")

for i, (name, depth) in enumerate(depth_maps.items()):
    disparity = depth_to_disparity(depth)
    
    axes[i + 1, 0].imshow(depth, cmap="plasma")
    axes[i + 1, 0].set_title(f"{name} Depth Map")
    
    axes[i + 1, 1].imshow(disparity, cmap="magma")
    axes[i + 1, 1].set_title(f"{name} Disparity")

    axes[i + 1, 2].imshow(input_image, alpha=0.6)
    axes[i + 1, 2].imshow(disparity, cmap="magma", alpha=0.4)
    axes[i + 1, 2].set_title(f"Overlay")

plt.tight_layout()
plt.show()

# Convert depth map to point cloud
def depth_to_point_cloud(depth_map, img):
    h, w = depth_map.shape
    fx, fy, cx, cy = w / 2, h / 2, w / 2, h / 2  # Approximate intrinsics

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
point_cloud = depth_to_point_cloud(first_depth_map, input_image)

o3d.visualization.draw_geometries([point_cloud])
