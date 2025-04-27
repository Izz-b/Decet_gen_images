###Construire un modèle capable de détecter si une image est réelle ou générée par une IA

---

# 🖼️ Detecting AI-Generated Images Using Vision Transformers (ViT)

## 📋 Project Overview

This project focuses on **classifying images as real or AI-generated** using a **Vision Transformer (ViT)** model.

With the rise of **diffusion models** and **GANs** for content generation, spotting **global inconsistencies** (like texture drift or unnatural object relations) is essential. Vision Transformers are particularly strong at this because they **capture long-range dependencies** across an image — better than CNNs, which focus on local features.

In this project, we:
- Load a dataset (`CIFAKE`) containing real and synthetic images.
- Preprocess images to match ViT input expectations.
- Fine-tune a pre-trained **ViT model** for binary classification (Real vs Fake).
- Train, evaluate, and perform inference on unseen images.
- Save and reload the trained model.

---

## 🛠️ Setup and Imports

```python
import os
import torch
import pathlib
import PIL
import cv2

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification, AutoImageProcessor
import torch.nn as nn
import torch.optim as optim
from PIL import Image
```

---

## 📂 Dataset Preparation

We use the **CIFAKE dataset** structured as:

```
/train
    /REAL
    /FAKE
/test
    /REAL
    /FAKE
```

We prepare smaller subsets for faster training:
- 20,000 samples for training.
- 5,000 samples for testing.

Transformations include resizing and normalization based on ViT's pre-trained settings.

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])
```

---

## 🧠 Model Setup

We fine-tune the **ViT-Base** model pre-trained on **ImageNet-21k**:

```python
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)
```

- Loss Function: `CrossEntropyLoss`
- Optimizer: `Adam` (learning rate = `5e-5`)
- Device: `GPU` if available

---

## 🔥 Training Loop

Training for **3 epochs**, we monitor loss every few steps.

```python
for epoch in range(num_epochs):
    model.train()
    for index, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        
        if index % 10 == 0:
            print(f"loss at {loss} at step :{index}")
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
```

---

## 📈 Evaluation

After training, we evaluate performance on the validation set:

```python
model.eval()
with torch.no_grad():
    ...
```

Metrics:
- **Validation Accuracy**: ~98–99%
- **Validation Loss**: Low (e.g., 0.02)

---

## 🔍 Inference on a Custom Image

Example of predicting on an unseen image:

```python
img = Image.open(image_path).convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

print("Predicted label:", model.config.id2label[predicted_class_idx])
```

---

## 💾 Model Saving and Loading

Save the trained model:

```python
torch.save(model.state_dict(), "isimage.pth")
```

Reload the model for future inference:

```python
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load("isimage.pth"))
model.to(device)
model.eval()
```

---

## 🎯 Why Does This Approach Work?

- **Vision Transformers** model global image features instead of just local patches.
- **Global inconsistencies** (e.g., strange texture flows, unnatural object relations) are **common in AI-generated images**, especially from diffusion models and GANs.
- Therefore, ViTs are particularly well-suited to detect fake images by **capturing these global clues**.

---

## ✅ Final Results

- **Training Loss**: 0.0204 after 3 epochs
- **Validation Accuracy**: ~98–99%

This indicates that the Vision Transformer effectively learned to distinguish between real and synthetic images with high reliability!

---

## 📌 Future Improvements

- Fine-tune larger models (ViT-Large, ViT-Huge).
- Use newer architectures (e.g., Swin Transformers, DINOv2).
- Train with more diverse synthetic datasets.

---

# 🚀 Conclusion

Vision Transformers offer a powerful method for detecting AI-generated images, leveraging their ability to **capture global features** and **long-range dependencies** that traditional CNNs often miss.  
This project demonstrates how transfer learning with ViT can lead to **highly accurate and practical models** for synthetic image detection!

