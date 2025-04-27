####Construire un modèle capable de détecter si une image est réelle ou générée par une IA 
📄 Detecting AI-Generated Images Using Vision Transformers (ViT)

✨ Project Overview

This notebook tackles the detection of AI-generated (fake) vs real images using a Vision Transformer (ViT) model.

With the rise of diffusion models and GANs in generating synthetic content, being able to capture global inconsistencies (such as unnatural textures or unrealistic structures) is key. Vision Transformers are excellent for this because they naturally model long-range dependencies across an image — much better than traditional CNNs that focus more on local features.

In this project, we:

Load a dataset (CIFAKE) consisting of real and AI-generated images.

Preprocess images suitable for ViT input.

Fine-tune a pre-trained Vision Transformer (google/vit-base-patch16-224-in21k) for binary classification (real vs fake).

Train and evaluate the model.

Test the model on an unseen image.

Save and reload the model for deployment.

🛠️ Setup and Imports

We import the essential libraries for:

Vision Transformer model loading and fine-tuning (from Huggingface transformers).

Dataset handling (torchvision.datasets, DataLoader).

Preprocessing (torchvision.transforms).

Utilities (OS handling, path management).

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

📂 Dataset Preparation

We work with the CIFAKE dataset, containing two classes:

REAL: authentic real-world photos.

FAKE: AI-generated synthetic images.

Data structure is assumed to be:

/train
    /REAL
    /FAKE
/test
    /REAL
    /FAKE

We create smaller subsets for faster experimentation:

20,000 samples for training.

5,000 samples for testing.

Images are resized and normalized using the pre-trained ViT image processor's settings.

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

🧠 Model Setup

We fine-tune the ViT-Base model, pre-trained on ImageNet-21k:

Load the model with num_labels=2 (Real or Fake).

Move it to GPU (if available).

Set the loss function as CrossEntropyLoss.

Use Adam optimizer with a learning rate of 5e-5.

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)

🔥 Training Loop

We train for 3 epochs, printing the loss every few steps to monitor training progress.

Each step involves:

Forward pass.

Calculating loss.

Backward pass (gradient calculation).

Optimizer step (parameter update).

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

📈 Evaluation

We evaluate the model on the validation set and report:

Validation Accuracy

Validation Loss

The final epoch result showed:

Epoch 3/3, Loss: 0.0204
Validation Accuracy: (around 98-99%)

Which indicates excellent learning and generalization for this task!

model.eval()
with torch.no_grad():
    ...

🔍 Inference on Custom Image

We also test the trained model on an unseen image.

Steps:

Load the image.

Preprocess with AutoImageProcessor.

Predict the class (REAL or FAKE).

inputs = processor(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

💾 Model Saving and Loading

Finally, we save the trained model weights and demonstrate how to load them again for future use:

torch.save(model.state_dict(), "isimage.pth")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load("isimage.pth"))
model.to(device)
model.eval()

🎯 Why Will This Work?

Vision Transformers (ViT) are exceptionally good at modeling global features in an image — not just small local textures.

Global inconsistencies such as texture drift, unrealistic structure, or bizarre spatial relations (common artifacts in diffusion- or GAN-based fake images) are captured naturally by ViTs.

Diffusion models, despite producing very detailed textures, often fail in global coherence — ViT can spot these global patterns better than traditional CNN-based classifiers.

Thus, this approach is highly suitable for detecting synthetic content generated by AI!

📌 Conclusion

This notebook successfully shows how Vision Transformers can be leveraged for detecting AI-generated vs real images with high accuracy.

Future improvements could include:

Training with larger datasets.

Fine-tuning using different ViT sizes (e.g., ViT-Large).

Experimenting with newer architectures like Swin Transformers or DINOv2.

