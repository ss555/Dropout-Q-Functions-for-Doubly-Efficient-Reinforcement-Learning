import torch
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained model
model = models.resnet50(pretrained=True)
# Remove the last layer (classification layer) to get features instead of predictions
model = torch.nn.Sequential(*(list(model.children())[:-1]))
print(model)
# Transform the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and transform an image
img = Image.open('fish.jpg').convert('RGB')
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Extract features
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    features = model(batch_t)

print(len(features))


# The 'features' tensor contains the extracted features of the image
