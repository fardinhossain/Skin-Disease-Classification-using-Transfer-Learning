"""
Skin Disease Classification using Transfer Learning
Models: GoogLeNet (Inception v1) and MobileNet V2
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
NUM_CLASSES = 8  # 8 skin disease classes

# Data directories
TRAIN_DIR = "dataset/train_set"
TEST_DIR = "dataset/test_set"

# Class names (based on folder structure)
CLASS_NAMES = [
    "BA-cellulitis",
    "BA-impetigo", 
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles"
]


def get_data_transforms():
    """Define data transformations for training and testing."""
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def load_datasets(train_transform, test_transform):
    """Load training and test datasets."""
    
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    return train_loader, test_loader, train_dataset.classes


def create_googlenet_model(num_classes, pretrained=True):
    """
    Create GoogLeNet (Inception v1) model for transfer learning.
    """
    # Load pretrained GoogLeNet
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Freeze early layers (optional - for fine-tuning)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Unfreeze the last few layers for fine-tuning
    for param in model.inception5a.parameters():
        param.requires_grad = True
    for param in model.inception5b.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def create_mobilenet_model(num_classes, pretrained=True):
    """
    Create MobileNet V2 model for transfer learning.
    """
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Unfreeze the last few layers for fine-tuning
    for param in model.features[-5:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Handle GoogLeNet auxiliary outputs during training
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Use main output
        
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels


def plot_training_history(history, model_name):
    """Plot training and validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax1.plot(history['test_acc'], label='Test Accuracy', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'{model_name} - Accuracy over Epochs')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history['train_loss'], label='Train Loss', marker='o')
    ax2.plot(history['test_loss'], label='Test Loss', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{model_name} - Loss over Epochs')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=150)
    plt.show()


def train_model(model, model_name, train_loader, test_loader, num_epochs, device):
    """Complete training loop for a model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use different learning rates for pretrained and new layers
    optimizer = optim.Adam([
        {'params': [p for p in model.parameters() if p.requires_grad]},
    ], lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(test_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, f'{model_name}_best.pth')
            print(f"✓ Best model saved with accuracy: {best_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    return model, history, predictions, labels


def compare_models(results):
    """Compare the performance of different models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Best Test Accuracy: {max(result['history']['test_acc']):.2f}%")
        print(f"  Final Train Accuracy: {result['history']['train_acc'][-1]:.2f}%")
        print(f"  Final Test Accuracy: {result['history']['test_acc'][-1]:.2f}%")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, result in results.items():
        ax.plot(result['history']['test_acc'], label=f'{model_name}', marker='o')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Model Comparison - Test Accuracy over Epochs')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()


def main():
    """Main function to train and evaluate models."""
    print("="*60)
    print("Skin Disease Classification using Transfer Learning")
    print("Models: GoogLeNet and MobileNet V2")
    print("="*60)
    
    # Get data transforms
    train_transform, test_transform = get_data_transforms()
    
    # Load datasets
    train_loader, test_loader, classes = load_datasets(train_transform, test_transform)
    
    # Store results for comparison
    results = {}
    
    # ============================================
    # Train GoogLeNet
    # ============================================
    print("\n" + "="*60)
    print("Creating GoogLeNet Model...")
    print("="*60)
    
    googlenet_model = create_googlenet_model(NUM_CLASSES, pretrained=True)
    print(f"GoogLeNet parameters: {sum(p.numel() for p in googlenet_model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in googlenet_model.parameters() if p.requires_grad):,}")
    
    googlenet_model, googlenet_history, googlenet_preds, googlenet_labels = train_model(
        googlenet_model, "GoogLeNet", train_loader, test_loader, NUM_EPOCHS, device
    )
    
    # Plot results for GoogLeNet
    plot_training_history(googlenet_history, "GoogLeNet")
    plot_confusion_matrix(googlenet_labels, googlenet_preds, classes, "GoogLeNet")
    print("\nGoogLeNet Classification Report:")
    print(classification_report(googlenet_labels, googlenet_preds, target_names=classes))
    
    results['GoogLeNet'] = {
        'history': googlenet_history,
        'predictions': googlenet_preds,
        'labels': googlenet_labels
    }
    
    # ============================================
    # Train MobileNet V2
    # ============================================
    print("\n" + "="*60)
    print("Creating MobileNet V2 Model...")
    print("="*60)
    
    mobilenet_model = create_mobilenet_model(NUM_CLASSES, pretrained=True)
    print(f"MobileNet V2 parameters: {sum(p.numel() for p in mobilenet_model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in mobilenet_model.parameters() if p.requires_grad):,}")
    
    mobilenet_model, mobilenet_history, mobilenet_preds, mobilenet_labels = train_model(
        mobilenet_model, "MobileNetV2", train_loader, test_loader, NUM_EPOCHS, device
    )
    
    # Plot results for MobileNet
    plot_training_history(mobilenet_history, "MobileNetV2")
    plot_confusion_matrix(mobilenet_labels, mobilenet_preds, classes, "MobileNetV2")
    print("\nMobileNet V2 Classification Report:")
    print(classification_report(mobilenet_labels, mobilenet_preds, target_names=classes))
    
    results['MobileNetV2'] = {
        'history': mobilenet_history,
        'predictions': mobilenet_preds,
        'labels': mobilenet_labels
    }
    
    # ============================================
    # Compare Models
    # ============================================
    compare_models(results)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nSaved files:")
    print("  - GoogLeNet_best.pth (Best GoogLeNet model)")
    print("  - MobileNetV2_best.pth (Best MobileNet model)")
    print("  - GoogLeNet_training_history.png")
    print("  - MobileNetV2_training_history.png")
    print("  - GoogLeNet_confusion_matrix.png")
    print("  - MobileNetV2_confusion_matrix.png")
    print("  - model_comparison.png")


if __name__ == "__main__":
    main()
