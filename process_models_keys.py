import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from networks import DepthDecoder, ResnetEncoder


### Load the encoder model
model_path = "models/mono_1024x320/fine_tuned/mse/"
encoder = ResnetEncoder(18, pretrained=True)  # Load a ResNet encoder with 18 layers
encoder.load_state_dict(torch.load(model_path + "encoder.pth"), strict=False)
state_dict = torch.load(model_path+"encoder.pth")

# Add missing keys
img_size = (640, 640)  # Image dimensions
toSave = encoder.state_dict()
toSave['height'] = img_size[0]
toSave['width'] = img_size[1]
toSave['use_stereo'] = False
print(toSave.keys())

torch.save(toSave, model_path+"encoder.pth")
print(f"Encoder model saved to {model_path}/encoder.pth")

### Load the depth decoder model
depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc)  # Load the depth decoder
depth_decoder.load_state_dict(torch.load(model_path + "depth.pth"))

torch.save(depth_decoder.state_dict(), model_path + "depth.pth")

