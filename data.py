import os
import json
from typing import List, Tuple
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_dir: str, annotations_file: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.annotations[idx]['file_name'])
        image = Image.open(img_name).convert('RGB')
        label = self.annotations[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(image_dir: str, annotations_file: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    full_dataset = ImageDataset(image_dir, annotations_file, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

# Usage example
if __name__ == "__main__":
    image_dir = ""
    annotations_file = ""
    train_loader, val_loader = get_data_loaders(image_dir, annotations_file)
    # print(f"Number of training batches: {len(train_loader)}")
    # print(f"Number of validation batches: {len(val_loader)}")