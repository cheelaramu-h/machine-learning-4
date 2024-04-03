from torchvision import models, transforms
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, GradientShap

# Ensure the correct device is used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

transform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Load the image
img_path = r'C:\Users\CHEELA RAMU HEMANTH\OneDrive\Desktop\daisy.jpg'  # Make sure to use the correct path to your image
img = Image.open(img_path)
transformed_img = transform(img)
input_tensor = transform_normalize(transformed_img).unsqueeze(0).to(DEVICE)

# Load the pretrained GoogLeNet model
model = models.googlenet(pretrained=True).to(DEVICE)
model.eval()

# Initialize attribution methods
integrated_gradients = IntegratedGradients(model)
gradient_shap = GradientShap(model)

# Set requires_grad attribute of tensor. Important for attribution
input_tensor.requires_grad = True

# Predict the class label
output = model(input_tensor)
pred_label = output.argmax(dim=1).item()

# Compute Integrated Gradients attribution
attr_ig = integrated_gradients.attribute(input_tensor, target=pred_label)
attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))

# Compute GradientShap attribution
baseline = torch.zeros_like(input_tensor).to(DEVICE)
attr_gs = gradient_shap.attribute(input_tensor, baselines=baseline, target=pred_label)
attr_gs = np.transpose(attr_gs.squeeze().cpu().detach().numpy(), (1, 2, 0))

# Visualize
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axs[0].imshow(transformed_img.permute(1, 2, 0))
axs[0].axis('off')
axs[0].set_title('Original Image')

# Integrated Gradients
# Summing attribution across color channels for better visualization
axs[1].imshow(np.sum(attr_ig, axis=2), cmap='hot')
axs[1].axis('off')
axs[1].set_title('Integrated Gradients')

# GradientShap
# Summing attribution across color channels for better visualization
axs[2].imshow(np.sum(attr_gs, axis=2), cmap='hot')
axs[2].axis('off')
axs[2].set_title('GradientShap')

plt.tight_layout()
plt.show()
