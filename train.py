import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification
from tqdm import tqdm
from data_preprocessor import get_data_loaders

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_accuracy = train_correct / train_total
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_accuracy = val_correct / val_total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    return model

# Usage example
if __name__ == "__main__":
    image_dir = ""
    annotations_file = ""
    train_loader, val_loader = get_data_loaders(image_dir, annotations_file)
    
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=len(train_loader.dataset.dataset.class_to_idx))
    trained_model = train_model(model, train_loader, val_loader)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), "vit_trained_model.pth")