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

# Load models (specify paths to pre-trained MonoDepth2 models)
MODEL_PATHS = {
    "MSE Train Original": "models/mono_1024x320/trained_env/dem/combined/u16__lindepth/", 
    "MSE Train Original2": "models/mono_1024x320/trained_env/dem/combined/u16__lindepth/" # Exammodels/mono_1024x320/fine_tuned/filtered_depth/cople path
    # "Depth Fourier Filtered": "mbined/normalized/",
    # "Depth-Dem Regularized": "models/mono_1024x320/fine_tuned/dem/combined/normalized/",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image
gt = True
index = 7
#image_path = f"datasets/atlas-tiny/image/opt_000{index}.png"  # Specify your image path
#image_path = "/home/stesilve/Documents/github/pivot/dfvo/dataset/moonLanding/01_8bit/000090.png"

image_path = "/users/mbussolino/Documents/Datasets/dataset_depth_00/imgs/frame-vis_00000055.png"
input_image = Image.open(image_path).convert('RGB')  # Open as RGB to ensure 3 channels (Monodepth2 expects this)

if gt:
    #depth_path = f"datasets/atlas-tiny/dem/dem_000{index}.png"  # Specify your depth map path
    depth_path = "/users/mbussolino/Documents/Datasets/dataset_depth_00/depth_maps_lin/depth_00055.png"
    depth_gt = Image.open(depth_path).convert('I;16')  # Depth maps are single-channel
#depth_gt = np.array(depth_gt).astype(np.uint16) / 65535.0 * 128.0 # Normalize to [0, 1]

# Preprocess image
def preprocess_image(img, width=640, height=640):
    #img = cv2.resize(img, (width, height))
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Resize((320,1024))
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
    print(input_tensor.shape)
    depth_maps[name] = infer_depth(name, encoder, depth_decoder, input_tensor)

# Convert depth to disparity
def depth_to_disparity(depth_map):
    return 1.0 / (depth_map + 1e-6)  # Avoid division by zero

# Plot results
if gt:
    print(len(depth_maps))
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
plt.savefig("test_sat.png")

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

def create_colormap_image_with_annotations(image_path, ax, title, num_annotations=10, resize_shape=None, selected_indices=None):
    # Load the image in unchanged mode (preserves 16-bit depth)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Error: Could not load image {image_path}.")
        return
    
    # Resize if specified
    if resize_shape is not None:
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_NEAREST)

    # Check if the image is 16-bit
    if img.dtype != np.uint16:
        print(f"Warning: The image {image_path} is not 16-bit grayscale.")
    
    cut_max = 65535
    cut_min = 0
    mask = (img < cut_max) & (img > cut_min)
    
    # Normalize for colormap
    norm_img = np.zeros_like(img, dtype=np.float32)
    norm_img[mask] = (img[mask] - cut_min) / (cut_max - cut_min)
    norm_img = np.clip(norm_img, 0, 1)
    
    # Apply colormap
    colormap = cm.viridis(norm_img)
    colormap_img = (colormap[:, :, :3] * 255).astype(np.uint8)
    
    # Plot
    ax.imshow(colormap_img)
    ax.set_title(title)
    ax.axis('off')
    
    # Annotate only masked points
    ys, xs = np.where(mask)
    if len(ys) == 0:
        print(f"No valid masked pixels found in {image_path}.")
        return

    #random.seed(42)  # You can choose any integer seed value
    if selected_indices is None:
        selected_indices = random.sample(range(len(ys)), min(num_annotations, len(ys)))
        for idx in selected_indices:
            y, x = ys[idx], xs[idx]
            value = img[y, x]/65535  # Normalize to [0, 1]
            m = 10
            M = 160
            #value = np.exp(value*(np.log(M)-np.log(m))+np.log(m))
            value = value*(M-m)+m
            ax.text(x, y, f"{value:.2f}", fontsize=6, color='white', ha='center', va='center', 
            bbox=dict(facecolor='black', alpha=0.4, lw=0))
    else:
        for y, x in zip(*selected_indices):
            #y, x = ys[idx], xs[idx]
            value = img[y, x]/65535  # Normalize to [0, 1]
            m = 10
            M = 160
            #value = np.exp(value*(np.log(M)-np.log(m))+np.log(m))
            value = value*(M-m)+m
            ax.text(x, y, f"{value:.2f}", fontsize=6, color='white', ha='center', va='center', 
            bbox=dict(facecolor='black', alpha=0.4, lw=0))
    return [ys[selected_indices], xs[selected_indices]]


# Generate and visualize 3D point cloud for the first model
first_depth_map = list(depth_maps.values())[0]
input_image = np.asarray(input_image)

# Salva la depth map
output_depth_path = "depth_output.png"  # Specifica il percorso di salvataggio
depth_map_to_save = depth_maps[name]  # Depth map per il modello corrente

# Normalizza la depth map e salvala come immagine
depth_normalized = np.uint16(depth_map_to_save / depth_map_to_save.max() * 65535)  # Normalizza i valori tra 0 e 65535
cv2.imwrite(output_depth_path, depth_normalized)

#point_cloud = depth_to_point_cloud(first_depth_map, input_image)

#o3d.visualization.draw_geometries([point_cloud])

# Example usage
gt_path = depth_path
test_path = output_depth_path

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
#random.seed(42)  # You can choose any integer seed value
idxs = create_colormap_image_with_annotations(gt_path, axs[0], title='Ground Truth', resize_shape=(1024, 320))
create_colormap_image_with_annotations(test_path, axs[1], title='Test Image', selected_indices=idxs)

plt.tight_layout()
plt.savefig("out_test.png", dpi=300) 