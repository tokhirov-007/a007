import os
import requests
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ddgs import DDGS
PARSE_IMAGES = False


from PIL import Image
from io import BytesIO

def download_images(query, folder, max_images=15):
    os.makedirs(folder, exist_ok=True)
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_images * 2)
        count = 0
        for result in results:
            if count >= max_images:
                break
            try:
                url = result["image"]
                ext = url.split(".")[-1].split("?")[0]
                if len(ext) > 5: ext = "jpg"

                img_data = requests.get(url, timeout=5).content
                img = Image.open(BytesIO(img_data))
                img.verify()

                with open(f"{folder}/{query}_{count}.{ext}", "wb") as f:
                    f.write(img_data)
                print(f"[✓] Saved: {folder}/{query}_{count}.{ext}")
                count += 1
            except Exception as e:
                print(f"[✗] Error ({url}): {e}")



categories = ["cat", "dog", "car", "airplane", "ship", "truck", "horse", "frog", "bird", "deer"]

if PARSE_IMAGES:
    print("📥 Начинаем парсинг...")
    for category in categories:
        download_images(category, f"dataset/{category}", max_images=15)
    print("✅ Парсинг завершён.\n")


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder("dataset", transform=transform)
trainloader = DataLoader(dataset, batch_size=16, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 32x32 → 16x16
        x = self.pool(torch.relu(self.conv2(x)))   # 16x16 → 8x8
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN(num_classes=len(categories)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🏋️ Обучение...")
for epoch in range(5):
    running_loss = 0.0
    correct, total = 0, 0
    model.train()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")
print("✅ Обучение завершено.\n")

torch.save(model.state_dict(), "parsed_cnn_model.pth")

dataiter = iter(trainloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

print("🎯 GroundTruth:", ' | '.join([dataset.classes[labels[j]] for j in range(8)]))
print("🔮 Predicted:  ", ' | '.join([dataset.classes[predicted[j]] for j in range(8)]))

img = torchvision.utils.make_grid(images[:8].cpu())
img = img / 2 + 0.5
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.axis("off")
plt.savefig("predictions.png")
print("🖼️ Сохранено как predictions.png")
