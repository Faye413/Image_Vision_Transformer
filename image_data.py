import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class AdvancedImageDataset(Dataset):
    def __init__(self, image_paths, labels, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        
        self.basic_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.augment:
            image = self.augment_transform(image)
        else:
            image = self.basic_transform(image)
        
        return image, label

class MixupAugmentation:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch
        batch_size = len(images)
        
        # Generate mixup weights
        weights = np.random.beta(self.alpha, self.alpha, batch_size)
        weights = np.maximum(weights, 1 - weights)
        weights = torch.from_numpy(weights).float().to(images.device)
        weights = weights.view(batch_size, 1, 1, 1)
        
        # Perform mixup
        indices = torch.randperm(batch_size)
        mixed_images = weights * images + (1 - weights) * images[indices]
        mixed_labels = weights.squeeze() * labels + (1 - weights.squeeze()) * labels[indices]
        
        return mixed_images, mixed_labels

# Usage example
if __name__ == "__main__":
    # Assume you have lists of image_paths and labels
    image_paths = [""]
    labels = []
    
    dataset = AdvancedImageDataset(image_paths, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    mixup = MixupAugmentation(alpha=0.2)
    
    for batch in dataloader:
        augmented_batch = mixup(batch)
        # Use augmented_batch for training