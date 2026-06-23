import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import time
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        return
        
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Define transforms (with light data augmentation for training)
    train_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomRotation(10),  # Slight rotation
        transforms.RandomAffine(0, translate=(0.05, 0.05)),  # Slight shifting
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 2. Load dataset
    full_dataset = datasets.ImageFolder(dataset_dir)
    num_classes = len(full_dataset.classes)
    print(f"Dataset classes: {full_dataset.classes}")
    print(f"Total images found: {len(full_dataset)}")
    
    if len(full_dataset) < 10:
        print("Error: Too few images to train. Please wait for the dataset extractor to gather more images.")
        return
        
    # Split into train and validation (85% train, 15% validation)
    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply specific transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 3. Load base model (ResNet50 with Chars74k pre-trained weights)
    base_checkpoint_path = os.path.join(script_dir, "resnet50_chars74k.pth")
    if not os.path.exists(base_checkpoint_path):
        print(f"Error: Pre-trained weights not found at {base_checkpoint_path}")
        return
        
    print(f"Loading pre-trained model from {base_checkpoint_path}...")
    base_checkpoint = torch.load(base_checkpoint_path, map_location=device)
    base_classes = base_checkpoint["classes"]
    
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(base_classes))
    
    # Load state dict
    sd = base_checkpoint["model_state_dict"]
    clean_sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    
    # Modify the classification head for the new number of classes
    print(f"Replacing classifier head from {len(base_classes)} classes to {num_classes} classes.")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Ensure all model parameters require gradients for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    # 4. Optimizer (optimize all parameters with a smaller learning rate) and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 5. Training loop
    epochs = 15
    best_acc = 0.0
    
    print("\n--- Starting Fine-tuning ---")
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += inputs.size(0)
            
        epoch_loss = running_loss / train_size
        epoch_train_acc = float(correct_train.item()) / train_size
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / val_size
        epoch_val_acc = correct_val.item() / val_size
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")
              
        # Save best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_weights_path = os.path.join(script_dir, "resnet50_finetuned.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": full_dataset.classes
            }, best_weights_path)
            print(f"--> Saved best model with validation accuracy: {best_acc:.4f}")
            
    print("\n--- Training Completed ---")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
