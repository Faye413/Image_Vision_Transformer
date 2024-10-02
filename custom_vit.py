import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import vit_b_16
from transformers import ViTConfig, ViTForImageClassification

class CustomViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CustomViT, self).__init__()
        
        # Load pretrained ViT
        if pretrained:
            self.vit = vit_b_16(pretrained=True)
        else:
            config = ViTConfig()
            self.vit = ViTForImageClassification(config)
        
        # Replace the classification head
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.heads.head.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Add attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(self.vit.hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Get the output from the ViT encoder
        features = self.vit.encoder(x)
        
        # Apply attention pooling
        attention_weights = self.attention_pool(features)
        pooled_features = torch.sum(features * attention_weights, dim=1)
        
        # Pass through the classification head
        logits = self.vit.heads(pooled_features)
        
        return logits

def train_custom_vit(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_accuracy = train_correct / train_total
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_accuracy = val_correct / val_total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        scheduler.step()
    
    return model

# Usage example
if __name__ == "__main__":
    num_classes = 10  # Replace with your number of classes
    model = CustomViT(num_classes)
    
    # Assume you have created train_loader and val_loader
    trained_model = train_custom_vit(model, train_loader, val_loader)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), "custom_vit_trained_model.pth")