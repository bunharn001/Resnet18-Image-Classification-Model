import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import os
from PIL import Image, ImageDraw, ImageFont

# Paths
data_dir = r"C:\Users\ss790\Documents\Python"
predict_dir = os.path.join(data_dir, "predict")
model_path = os.path.join(data_dir, "resnet18_dogcat.pth")
result_dir = os.path.join(data_dir, "result")

# Classes (must match your training order)
classes = ['cat', 'dog']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model setup
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Ensure result directory exists
os.makedirs(result_dir, exist_ok=True)

# Font for label (larger size)
try:
    font = ImageFont.truetype("arial.ttf", 96)  # Increased font size
except:
    font = ImageFont.load_default()

# Counters for renaming
counters = {cls: 1 for cls in classes}

# Inference loop
for fname in os.listdir(predict_dir):
    fpath = os.path.join(predict_dir, fname)
    if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # Load and preprocess image
        img = Image.open(fpath).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            class_idx = predicted.item()
            class_name = classes[class_idx]

        # Draw label in the center of the image
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        text = class_name
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        img_width, img_height = img_draw.size
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
        margin = 10
        draw.rectangle(
            [x - margin, y - margin, x + text_width + margin, y + text_height + margin],
            fill=(255, 255, 255, 180)
        )
        draw.text((x, y), text, fill=(255, 0, 0), font=font)

        # Rename and save to result directory
        count = counters[class_name]
        new_fname = f"{class_name}_{count:03d}.jpg"
        counters[class_name] += 1
        result_path = os.path.join(result_dir, new_fname)
        img_draw.save(result_path)

        # Delete original image
        os.remove(fpath)

        print(f"{fname} → {new_fname}: {class_name}")

print("Inference complete. Labeled and renamed images saved in 'result' folder.")