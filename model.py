# model_runner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

classes = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']  
model = SimpleCNN(num_classes=len(classes))
model.load_state_dict(torch.load("simple_cnn.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

def run(input: dict) -> dict:
    image_data = input["image"]
    image = Image.open(image_data)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()
        predicted_class = torch.argmax(outputs, 1).item()

    return {
        "prediction": classes[predicted_class],
        "probabilities": {classes[i]: round(probabilities[i], 3) for i in range(len(classes))}
    }
