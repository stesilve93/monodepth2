import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from networks import DepthDecoder, ResnetEncoder
from utils import normalize_image
from datasets.ml_dataset import DepthDataset, ScaleInvariantLoss, EdgeLoss, SSIMLoss, CombinedLoss
from torchvision import transforms
import os

from torch.utils.tensorboard import SummaryWriter

# Paths
source_depth = "filtered_depth"  # Source of depth maps ["dem", "depth"]
img_dir = "datasets/atlas-tiny/image/"  # Directory containing input images
depth_dir = "datasets/atlas-tiny/"+source_depth  # Directory containing ground truth depth maps
model_path = "models/mono_1024x320/"  # Path to pre-trained model weights
loss = "mse"  # Loss function to use ["scale_invariant", "mse"]
log_dir = "runs/fine_tuning/"+source_depth+"/"+loss  # Directory for TensorBoard logs
save_path = "fine_tuned/"+source_depth+"/"+loss  # Directory to save the fine-tuned model

# Hyperparameters
batch_size = 4  # Number of samples per batch
learning_rate = 1e-5  # Learning rate for the optimizer
num_epochs = 100  # Number of training epochs
img_size = (640, 640)  # Image dimensions
early_stopping_patience = 15  # Stop if no improvement for tot epochs
best_val_loss = float("inf")
patience_counter = 0  # Counter for early stopping

# Full dataset (this is the entire dataset, no split yet)
full_dataset = DepthDataset(img_dir, depth_dir, img_size=img_size, source=source_depth)

# Split dataset into 80% train, 10% validation, 10% test
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Create the splits
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoaders for each subset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load Monodepth2 Model
encoder = ResnetEncoder(18, pretrained=True)  # Load a ResNet encoder with 18 layers
depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc)  # Load the depth decoder

# Load pre-trained weights
encoder.load_state_dict(torch.load(model_path + "encoder.pth", weights_only=True), strict=False)
depth_decoder.load_state_dict(torch.load(model_path + "depth.pth", weights_only=True))

# Set to training mode
encoder.train()
depth_decoder.train()

# Freeze layers (optional)
# Uncomment and adjust these lines to freeze specific layers in the encoder
# for param in encoder.parameters():
#      param.requires_grad = False

# # Optionally, unfreeze specific layers (e.g., the last few layers)
# for param in encoder.encoder.layer4.parameters():
#      param.requires_grad = True

# Now, only the layers that are not frozen will be trained (e.g., the decoder and unfrozen encoder layers)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
depth_decoder = depth_decoder.to(device)

# Loss function
if loss == "scale_invariant":
    loss_fn = ScaleInvariantLoss()
elif loss == "mse":
    loss_fn = nn.MSELoss()  # Mean Squared Error for depth supervision
elif loss == "edge":
    loss_fn = EdgeLoss()  # Edge-aware depth loss
elif loss == "ssim":
    loss_fn = SSIMLoss()  # Structural Similarity Index loss
elif loss == "combined":
    loss_fn = CombinedLoss()  # Combined loss (scale-invariant + edge-aware)

# Optimizer
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(depth_decoder.parameters()), lr=learning_rate)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# TensorBoard writer
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

# Helper function for validation/testing
def evaluate_model(loader, encoder, depth_decoder, loss_fn, device):
    encoder.eval()  # Set encoder to evaluation mode
    depth_decoder.eval()  # Set decoder to evaluation mode
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    count = 0

    with torch.no_grad():  # Disable gradient computation
        for batch in loader:
            images = batch["image"].to(device)
            gt_depth = batch["depth"].to(device)

            features = encoder(images)
            outputs = depth_decoder(features)
            pred_depth = outputs[("disp", 0)]

            loss = loss_fn(pred_depth, gt_depth)
            total_loss += loss.item()

            mae = torch.mean(torch.abs(pred_depth - gt_depth))
            rmse = torch.sqrt(torch.mean((pred_depth - gt_depth) ** 2))

            total_mae += mae.item()
            total_rmse += rmse.item()
            count += 1

    avg_loss = total_loss / count
    avg_mae = total_mae / count
    avg_rmse = total_rmse / count

    return avg_loss, avg_mae, avg_rmse

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    encoder.train()  # Set encoder to training mode
    depth_decoder.train()  # Set decoder to training mode

    # Get current learning rate
    print("--------------------------------")
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{num_epochs}, Learning Rate: {current_lr}")

    for step, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        gt_depth = batch["depth"].to(device)

        # Forward pass
        features = encoder(images)
        outputs = depth_decoder(features)
        
        # Monodepth2 outputs a dictionary; get the predicted depth
        pred_depth = outputs[("disp", 0)]  # Output at the highest resolution

        # Compute supervised loss
        loss = loss_fn(pred_depth, gt_depth)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Log training loss
        if step % 10 == 0:
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + step)

        # Log images every 100 steps
        if step % 100 == 0:
            writer.add_images("Train/Input Images", images, epoch)
            writer.add_images("Train/Predicted Depth", pred_depth, epoch)
            writer.add_images("Train/Ground Truth Depth", gt_depth, epoch)

    # Validation
    val_loss, val_mae, val_rmse = evaluate_model(val_loader, encoder, depth_decoder, loss_fn, device)
    writer.add_scalar("Validation/Loss", val_loss, epoch)
    writer.add_scalar("Validation/MAE", val_mae, epoch)
    writer.add_scalar("Validation/RMSE", val_rmse, epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(train_loader)}, Val Loss: {val_loss}")

    # Test evaluation
    test_loss, test_mae, test_rmse = evaluate_model(test_loader, encoder, depth_decoder, loss_fn, device)
    writer.add_scalar("Test/Loss", test_loss, epoch)
    writer.add_scalar("Test/MAE", test_mae, epoch)
    writer.add_scalar("Test/RMSE", test_rmse, epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss}, Test MAE: {test_mae}, Test RMSE: {test_rmse}")

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement. Patience counter: {patience_counter}")

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered. Training terminated.")
        break

    # Step the learning rate scheduler
    scheduler.step()

# Check if the folder exists
if not os.path.exists(model_path+save_path):
    # Create the folder if it doesn't exist
    os.makedirs(model_path+save_path)
    print(f"Folder created at {model_path+save_path}")

# Save fine-tuned model
toSave = encoder.state_dict()
toSave['height'] = img_size[0]
toSave['width'] = img_size[1]
toSave['use_stereo'] = False
torch.save(toSave, model_path+save_path+"/encoder.pth")
print(f"Encoder model saved to {model_path+save_path}/encoder.pth")

torch.save(depth_decoder.state_dict(), model_path+save_path+"/depth.pth")
print(f"Decoder model saved to {model_path+save_path}/depth.pth")

# Close TensorBoard writer
writer.close()
