import torch
from torchviz import make_dot
from networks import ResnetEncoder  # Assuming ResnetEncoder is your encoder class

# Initialize the encoder
encoder = ResnetEncoder(18, pretrained=True)

# Set to evaluation mode (optional for visualization)
encoder.eval()

# Create a dummy input (batch size=1, channels=3, height=640, width=640)
dummy_input = torch.randn(1, 3, 640, 640)

# Perform a forward pass
features = encoder(dummy_input)

# Generate a visualization graph
# Pass the first feature map from the encoder (just for demonstration)
dot = make_dot(features[-1], params=dict(encoder.named_parameters()),show_attrs=True, show_saved=True)

# Save the graph as a file
dot.format = "png"  # You can change this to 'pdf' or other supported formats
dot.render("encoder_visualization")  # Saves as "encoder_visualization.png"

# To display inline (if using a Jupyter notebook)
# from IPython.display import Image
# Image(filename="encoder_visualization.png")
