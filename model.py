# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 08:41:18 2024

@author: BOUROU-PC
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import matplotlib.pyplot as plt
import seaborn as sns

# === DATASET ===
class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.root_dir, f"{row['patientId']}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(row['Target'], dtype=torch.float32)
    
        if self.transforms:
            image = self.transforms(image)
    
        # Convert metadata to tensor
        metadata = torch.tensor([
            row['age'],
            1.0 if row['sex'] == 'M' else 0.0,
            1.0 if row['position'] == 'AP' else 0.0
        ], dtype=torch.float32)
        return image, label, metadata

# === TRANSFORMS ===
# Enhanced Data Augmentation
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = train_transforms

# === MODEL ===
class EnhancedPneumoniaModel(nn.Module):
    def __init__(self, pretrained=True):
        super(EnhancedPneumoniaModel, self).__init__()
        # Load EfficientNet
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = self.backbone.features(torch.rand(1, 3, 224, 224)).flatten(1).shape[1]

        # Modify the classifier with additional layers
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features + 3, 512),  # Include metadata features
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x, metadata):
        features = self.backbone.features(x)
        features = torch.flatten(features, 1)
        combined = torch.cat((features, metadata), dim=1)
        return self.backbone.classifier(combined)

# === TRAINING AND VALIDATION ===
def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, patience=3):
    start_time = time.time()
    loss_fn = nn.BCEWithLogitsLoss()
    model = model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    train_accuracies, val_accuracies, val_losses = [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels, metadata in train_loader:
            images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)

            preds = model(images, metadata).squeeze()
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(preds) > 0.4).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, labels, metadata in val_loader:
                images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)

                preds = model(images, metadata).squeeze()
                loss = loss_fn(preds, labels)
                val_loss += loss.item()

                predicted = (torch.sigmoid(preds) > 0.4).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(torch.sigmoid(preds).cpu().numpy())

        val_accuracy = correct_val / total_val
        val_auc = roc_auc_score(all_labels, all_preds)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Plot Accuracy and Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Development')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss Development')

    plt.tight_layout()
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    correct_val = 0
    total_val = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels, metadata in val_loader:
            images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)

            preds = model(images, metadata).squeeze()
            loss = loss_fn(preds, labels)
            val_loss += loss.item()

            predicted = (torch.sigmoid(preds) > 0.4).float()
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(preds).cpu().numpy())

    val_accuracy = correct_val / total_val
    auc_score = roc_auc_score(all_labels, all_preds)
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, AUC: {auc_score:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    return val_accuracy

# === TEST SINGLE IMAGE ===
def test_single_image(model, dataset, index, device):
    model.eval()
    image, label, metadata = dataset[index]
    # Display metadata
    print("Patient Metadata:")
    print(metadata)
    with torch.no_grad():
        image_tensor = image.to(device).unsqueeze(0)
        metadata_tensor = metadata.to(device).unsqueeze(0)
        output = model(image_tensor, metadata_tensor).squeeze()
        prediction = torch.sigmoid(output).item()

    # Denormalize the image for display
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy
    image_np = (image_np * std.numpy() + mean.numpy()).clip(0, 1)  # Denormalize and clip

    # Display image
    plt.imshow(image_np)
    plt.title(f"Label: {label.item()}, Prediction: {prediction:.4f}")
    plt.axis('off')
    plt.show()



# === MAIN ===
if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # File paths
    dataset_path = "stage2_train_metadata.csv"
    images_path = "images/"

    # Load and preprocess data
    annotations = pd.read_csv(dataset_path)
    annotations.drop(['x', 'y', 'width', 'height', 'modality'], axis=1, inplace=True)
    annotations.drop_duplicates(inplace=True)
    
    # Data Overview
    print("Total rows in train_labels:", annotations.shape[0])
    print("Total unique patients:", annotations['patientId'].nunique())

    # Class Distribution Visualization
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Target', data=annotations)
    plt.title("Target Class Distribution")
    plt.show()

    print("Loading dataset...")
    print("Dataset loaded and duplicates removed.")
    print("Splitting data into train and validation sets...")

    train_df, val_df = train_test_split(annotations, test_size=0.2, random_state=123)
    train_df.to_csv("train_split.csv", index=False)
    val_df.to_csv("val_split.csv", index=False)

    print("Data split complete.")
    print("Initializing datasets and dataloaders...")

    # Load datasets
    train_dataset = PneumoniaDataset("train_split.csv", images_path, transforms=train_transforms)
    val_dataset = PneumoniaDataset("val_split.csv", images_path, transforms=val_transforms)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=6)

    print("Datasets and dataloaders initialized.")
    print("Initializing model...")

    # Initialize model, optimizer, and scheduler
    model = EnhancedPneumoniaModel(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)

    print("Model initialized.")
    print("Starting training process...")
    train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=7, device=device)

    # Save model
    torch.save(model.state_dict(), "enhanced_pneumonia_detection_model.pth")
    print("Model training complete and saved.")

    # Test single image from validation set
    print("Testing single image from validation set...")
    test_single_image(model, val_dataset, index=0, device=device)
